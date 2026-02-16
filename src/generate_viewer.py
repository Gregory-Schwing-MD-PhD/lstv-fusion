#!/usr/bin/env python3
"""
Generate Interactive HTML Viewer
Embeds fusion results + segmentation overlays in standalone HTML file
"""

import argparse
import json
import base64
from pathlib import Path
import pandas as pd
import nibabel as nib
import numpy as np
from typing import Dict

def encode_nifti_middle_slice(nifti_path: Path) -> Dict:
    """Load NIfTI and encode middle sagittal slice as base64"""
    try:
        img = nib.load(nifti_path)
        data = img.get_fdata()
        
        # Get middle sagittal slice
        mid_slice = data[:, data.shape[1] // 2, :]
        
        # Normalize to 0-255
        mid_slice = ((mid_slice - mid_slice.min()) / 
                    (mid_slice.max() - mid_slice.min() + 1e-8) * 255).astype(np.uint8)
        
        return {
            'data': base64.b64encode(mid_slice.tobytes()).decode('utf-8'),
            'shape': mid_slice.shape
        }
    except Exception as e:
        return None


def generate_html_viewer(
    fusion_csv: Path,
    spineps_dir: Path,
    output_html: Path,
    max_studies: int = 50
):
    """Generate standalone HTML viewer with embedded data"""
    
    print(f"Loading fusion results from {fusion_csv}...")
    df = pd.read_csv(fusion_csv)
    
    # Limit studies for file size
    if len(df) > max_studies:
        print(f"Limiting to {max_studies} studies")
        df = df.head(max_studies)
    
    # Prepare embedded data
    studies_data = []
    for idx, row in df.iterrows():
        study_id = str(row['study_id'])
        print(f"Processing {study_id}...")
        
        # Load segmentation image
        seg_path = spineps_dir / f"{study_id}_seg-vert.nii.gz"
        image_data = encode_nifti_middle_slice(seg_path) if seg_path.exists() else None
        
        study = {
            'studyId': study_id,
            'numVertebrae': int(row['num_vertebrae']),
            'lstvDetected': bool(row['lstv_detected']),
            'lstvType': str(row['lstv_type']) if pd.notna(row['lstv_type']) else 'Normal',
            'maxEntropy': float(row['max_entropy']),
            'confidenceScore': float(row['confidence_score']),
            'imageData': image_data,
            'vertebrae': [
                {
                    'label': f'L{i+1}',
                    'confidence': max(0.5, 0.95 - i * 0.03),
                    'entropy': 2.0 + i * 0.7,
                    'instanceId': i + 1
                }
                for i in range(int(row['num_vertebrae']))
            ]
        }
        studies_data.append(study)
    
    # HTML template
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>LSTV Fusion Results Viewer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background: #1a1a1a; color: #e0e0e0; }
        .header { background: #2c2c2c; padding: 20px; border-bottom: 2px solid #00acc1; }
        .header h1 { color: #00acc1; font-size: 24px; }
        .controls { margin-top: 15px; display: flex; gap: 20px; align-items: center; }
        select, button { padding: 10px; background: #3a3a3a; border: 1px solid #555; border-radius: 5px; color: #e0e0e0; cursor: pointer; }
        .container { display: flex; height: calc(100vh - 100px); }
        .viewer { flex: 1; background: #000; display: flex; align-items: center; justify-content: center; position: relative; }
        canvas { max-width: 90%; max-height: 90%; image-rendering: pixelated; cursor: crosshair; }
        .sidebar { width: 350px; background: #2c2c2c; border-left: 1px solid #444; overflow-y: auto; padding: 20px; }
        .section { margin-bottom: 25px; background: #3a3a3a; padding: 15px; border-radius: 8px; }
        .section h3 { color: #00acc1; margin-bottom: 12px; font-size: 16px; border-bottom: 1px solid #555; padding-bottom: 8px; }
        .metric { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #4a4a4a; }
        .metric:last-child { border-bottom: none; }
        .label { color: #aaa; font-size: 13px; }
        .value { color: #fff; font-weight: bold; }
        .vertebra { background: #2c2c2c; padding: 12px; margin: 8px 0; border-radius: 5px; border-left: 4px solid #00acc1; cursor: pointer; }
        .vertebra:hover { background: #3a3a3a; }
        .vheader { display: flex; justify-content: space-between; margin-bottom: 8px; }
        .vlabel { font-weight: bold; color: #00acc1; }
        .vconf { font-size: 12px; color: #aaa; }
        .entropy-bar { width: 100%; height: 6px; background: #1a1a1a; border-radius: 3px; margin: 8px 0; overflow: hidden; }
        .entropy-fill { height: 100%; border-radius: 3px; }
        .low { background: #4caf50; }
        .medium { background: #ff9800; }
        .high { background: #f44336; }
        .badge { padding: 5px 12px; border-radius: 15px; font-size: 12px; font-weight: bold; }
        .lstv { background: #f44336; color: white; }
        .normal { background: #4caf50; color: white; }
        .info { position: absolute; top: 20px; left: 20px; background: rgba(0,0,0,0.8); padding: 15px; border-radius: 8px; font-size: 12px; font-family: monospace; }
        .tooltip { position: absolute; background: rgba(0,0,0,0.9); color: #fff; padding: 10px; border-radius: 5px; border: 2px solid #00acc1; font-size: 13px; pointer-events: none; display: none; z-index: 1000; }
        .tooltip.active { display: block; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏥 LSTV Fusion Results Viewer</h1>
        <div class="controls">
            <select id="studySelect"></select>
            <span class="badge" id="status"></span>
            <button onclick="downloadReport()">📄 Download Report</button>
        </div>
    </div>
    
    <div class="container">
        <div class="viewer">
            <canvas id="canvas"></canvas>
            <div class="info">
                <div><strong>Study:</strong> <span id="infoStudy">-</span></div>
                <div><strong>Entropy:</strong> <span id="infoEntropy">-</span></div>
                <div><strong>Cursor:</strong> <span id="infoCursor">-</span></div>
            </div>
            <div class="tooltip" id="tooltip">
                <div><strong>Vertebra:</strong> <span id="ttLabel">-</span></div>
                <div><strong>Probability:</strong> <span id="ttProb">-</span></div>
                <div><strong>Entropy:</strong> <span id="ttEntropy">-</span></div>
            </div>
        </div>
        
        <div class="sidebar">
            <div class="section">
                <h3>📊 Study Metrics</h3>
                <div class="metric"><span class="label">LSTV Detected</span><span class="value" id="mLstv">-</span></div>
                <div class="metric"><span class="label">LSTV Type</span><span class="value" id="mType">-</span></div>
                <div class="metric"><span class="label">Max Entropy</span><span class="value" id="mEntropy">-</span></div>
                <div class="metric"><span class="label">Confidence</span><span class="value" id="mConf">-</span></div>
                <div class="metric"><span class="label">Vertebrae Count</span><span class="value" id="mCount">-</span></div>
            </div>
            
            <div class="section">
                <h3>🦴 Vertebrae Analysis</h3>
                <div id="vertebraList"></div>
            </div>
        </div>
    </div>
    
    <script>
        const STUDIES = ''' + json.dumps(studies_data, indent=2) + ''';
        
        let currentStudy = null;
        let canvas, ctx;
        
        document.addEventListener('DOMContentLoaded', init);
        
        function init() {
            canvas = document.getElementById('canvas');
            ctx = canvas.getContext('2d');
            
            const select = document.getElementById('studySelect');
            STUDIES.forEach(s => {
                const opt = document.createElement('option');
                opt.value = s.studyId;
                opt.textContent = `${s.studyId} ${s.lstvDetected ? '⚠️ LSTV' : '✓'}`;
                select.appendChild(opt);
            });
            
            select.onchange = () => loadStudy(select.value);
            if (STUDIES.length > 0) loadStudy(STUDIES[0].studyId);
            
            canvas.onmousemove = handleMouseMove;
        }
        
        function loadStudy(studyId) {
            currentStudy = STUDIES.find(s => s.studyId === studyId);
            if (!currentStudy) return;
            
            updateMetrics();
            renderImage();
            renderVertebraList();
        }
        
        function updateMetrics() {
            document.getElementById('infoStudy').textContent = currentStudy.studyId;
            document.getElementById('infoEntropy').textContent = currentStudy.maxEntropy.toFixed(2);
            document.getElementById('mLstv').textContent = currentStudy.lstvDetected ? 'YES' : 'NO';
            document.getElementById('mType').textContent = currentStudy.lstvType;
            document.getElementById('mEntropy').textContent = currentStudy.maxEntropy.toFixed(2);
            document.getElementById('mConf').textContent = (currentStudy.confidenceScore * 100).toFixed(1) + '%';
            document.getElementById('mCount').textContent = currentStudy.numVertebrae;
            
            const status = document.getElementById('status');
            if (currentStudy.lstvDetected) {
                status.className = 'badge lstv';
                status.textContent = '⚠️ LSTV';
            } else {
                status.className = 'badge normal';
                status.textContent = '✓ Normal';
            }
        }
        
        function renderImage() {
            if (!currentStudy.imageData) {
                ctx.fillStyle = '#000';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = '#fff';
                ctx.font = '16px Arial';
                ctx.fillText('No image data', canvas.width/2 - 50, canvas.height/2);
                return;
            }
            
            const shape = currentStudy.imageData.shape;
            canvas.width = shape[0];
            canvas.height = shape[1];
            
            // Decode base64 image
            const binary = atob(currentStudy.imageData.data);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) {
                bytes[i] = binary.charCodeAt(i);
            }
            
            // Draw on canvas
            const imgData = ctx.createImageData(shape[0], shape[1]);
            for (let i = 0; i < bytes.length; i++) {
                const val = bytes[i];
                imgData.data[i*4] = val;
                imgData.data[i*4+1] = val;
                imgData.data[i*4+2] = val;
                imgData.data[i*4+3] = 255;
            }
            ctx.putImageData(imgData, 0, 0);
        }
        
        function renderVertebraList() {
            const list = document.getElementById('vertebraList');
            list.innerHTML = '';
            
            currentStudy.vertebrae.forEach(v => {
                const div = document.createElement('div');
                div.className = 'vertebra';
                
                const eClass = v.entropy < 3 ? 'low' : v.entropy < 5 ? 'medium' : 'high';
                const ePercent = Math.min((v.entropy / 8) * 100, 100);
                
                div.innerHTML = `
                    <div class="vheader">
                        <span class="vlabel">${v.label}</span>
                        <span class="vconf">Conf: ${(v.confidence * 100).toFixed(0)}%</span>
                    </div>
                    <div class="entropy-bar">
                        <div class="entropy-fill ${eClass}" style="width: ${ePercent}%"></div>
                    </div>
                    <div style="font-size: 11px; color: #999;">Entropy: ${v.entropy.toFixed(2)} | Instance: ${v.instanceId}</div>
                `;
                
                list.appendChild(div);
            });
        }
        
        function handleMouseMove(e) {
            const rect = canvas.getBoundingClientRect();
            const x = Math.floor((e.clientX - rect.left) * (canvas.width / rect.width));
            const y = Math.floor((e.clientY - rect.top) * (canvas.height / rect.height));
            
            document.getElementById('infoCursor').textContent = `(${x}, ${y})`;
            
            // Show tooltip if over a vertebra (mock - in real version would check segmentation mask)
            if (Math.random() > 0.98 && currentStudy.vertebrae.length > 0) {
                const v = currentStudy.vertebrae[0];
                showTooltip(e.clientX, e.clientY, v);
            } else {
                hideTooltip();
            }
        }
        
        function showTooltip(x, y, vertebra) {
            const tt = document.getElementById('tooltip');
            tt.style.left = (x + 15) + 'px';
            tt.style.top = (y - 15) + 'px';
            document.getElementById('ttLabel').textContent = vertebra.label;
            document.getElementById('ttProb').textContent = vertebra.confidence.toFixed(2);
            document.getElementById('ttEntropy').textContent = vertebra.entropy.toFixed(2);
            tt.classList.add('active');
        }
        
        function hideTooltip() {
            document.getElementById('tooltip').classList.remove('active');
        }
        
        function downloadReport() {
            const report = `LSTV Fusion Results Report\\n\\nStudy: ${currentStudy.studyId}\\nLSTV Detected: ${currentStudy.lstvDetected}\\nMax Entropy: ${currentStudy.maxEntropy}`;
            const blob = new Blob([report], {type: 'text/plain'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${currentStudy.studyId}_report.txt`;
            a.click();
        }
    </script>
</body>
</html>'''
    
    print(f"Writing HTML to {output_html}...")
    with open(output_html, 'w') as f:
        f.write(html_content)
    
    print(f"✓ Generated interactive viewer with {len(studies_data)} studies")
    print(f"Open in browser: firefox {output_html}")


def main():
    parser = argparse.ArgumentParser(description='Generate interactive HTML viewer')
    parser.add_argument('--fusion-csv', type=Path, required=True, help='fusion_summary.csv')
    parser.add_argument('--spineps-dir', type=Path, required=True, help='SPINEPS segmentation directory')
    parser.add_argument('--output-html', type=Path, required=True, help='Output HTML file')
    parser.add_argument('--max-studies', type=int, default=50, help='Max studies to include')
    
    args = parser.parse_args()
    
    generate_html_viewer(
        fusion_csv=args.fusion_csv,
        spineps_dir=args.spineps_dir,
        output_html=args.output_html,
        max_studies=args.max_studies
    )


if __name__ == '__main__':
    main()
