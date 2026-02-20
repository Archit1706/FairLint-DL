export function getStyles(): string {
    return `
        :root {
            --bg-primary: #1e1e1e;
            --bg-secondary: #252526;
            --bg-tertiary: #2d2d30;
            --bg-hover: #3c3c3c;
            --text-primary: #e4e4e4;
            --text-secondary: #a0a0a0;
            --text-muted: #6e6e6e;
            --accent-blue: #4fc3f7;
            --accent-green: #4caf50;
            --accent-yellow: #ffc107;
            --accent-orange: #ff9800;
            --accent-red: #f44336;
            --accent-purple: #9c27b0;
            --border-color: #3c3c3c;
            --shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            --transition: all 0.2s ease;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 24px;
            max-width: 1400px;
            margin: 0 auto;
        }

        .header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 32px;
            padding-bottom: 24px;
            border-bottom: 1px solid var(--border-color);
        }

        .header-left h1 { font-size: 28px; font-weight: 600; color: var(--text-primary); margin-bottom: 8px; }
        .header-meta { display: flex; gap: 24px; color: var(--text-secondary); font-size: 14px; }
        .header-meta span { display: flex; align-items: center; gap: 6px; }

        .score-card {
            background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
            border-radius: 16px; padding: 24px; text-align: center; min-width: 200px;
            border: 1px solid var(--border-color);
        }
        .score-value { font-size: 48px; font-weight: 700; margin-bottom: 4px; }
        .score-label { font-size: 14px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; }
        .score-status { margin-top: 12px; padding: 6px 16px; border-radius: 20px; font-size: 13px; font-weight: 500; }

        .status-good { background: rgba(76, 175, 80, 0.2); color: var(--accent-green); }
        .status-warning { background: rgba(255, 152, 0, 0.2); color: var(--accent-orange); }
        .status-danger { background: rgba(244, 67, 54, 0.2); color: var(--accent-red); }

        .section { margin-bottom: 32px; animation: fadeIn 0.4s ease forwards; }
        .section:nth-child(2) { animation-delay: 0.1s; }
        .section:nth-child(3) { animation-delay: 0.2s; }
        .section:nth-child(4) { animation-delay: 0.3s; }
        .section:nth-child(5) { animation-delay: 0.4s; }

        .section-header { display: flex; align-items: center; gap: 12px; margin-bottom: 16px; }
        .section-icon {
            width: 32px; height: 32px; border-radius: 8px;
            display: flex; align-items: center; justify-content: center; font-size: 16px;
        }
        .section-title { font-size: 20px; font-weight: 600; color: var(--text-primary); }

        .cards-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }

        .card {
            background: var(--bg-secondary); border: 1px solid var(--border-color);
            border-radius: 12px; padding: 20px; transition: var(--transition);
        }
        .card:hover { background: var(--bg-tertiary); transform: translateY(-2px); box-shadow: var(--shadow); }

        .card-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px; }
        .card-label { font-size: 13px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; }
        .card-badge { padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; text-transform: uppercase; }

        .badge-success { background: rgba(76, 175, 80, 0.2); color: var(--accent-green); }
        .badge-warning { background: rgba(255, 152, 0, 0.2); color: var(--accent-orange); }
        .badge-danger { background: rgba(244, 67, 54, 0.2); color: var(--accent-red); }
        .badge-info { background: rgba(79, 195, 247, 0.2); color: var(--accent-blue); }

        .card-value { font-size: 32px; font-weight: 700; color: var(--text-primary); margin-bottom: 4px; }
        .card-unit { font-size: 16px; font-weight: 400; color: var(--text-secondary); margin-left: 4px; }
        .card-description { font-size: 13px; color: var(--text-muted); margin-top: 8px; }

        .progress-container { margin-top: 12px; }
        .progress-bar { height: 6px; background: var(--bg-primary); border-radius: 3px; overflow: hidden; }
        .progress-fill { height: 100%; border-radius: 3px; transition: width 0.3s ease; }
        .progress-labels { display: flex; justify-content: space-between; margin-top: 4px; font-size: 11px; color: var(--text-muted); }

        .chart-container {
            background: var(--bg-secondary); border: 1px solid var(--border-color);
            border-radius: 12px; padding: 20px; margin-top: 16px;
        }
        .chart-title { font-size: 16px; font-weight: 600; margin-bottom: 16px; color: var(--text-primary); }

        .neuron-list { display: flex; flex-direction: column; gap: 8px; }
        .neuron-item {
            display: flex; justify-content: space-between; align-items: center;
            padding: 12px 16px; background: var(--bg-tertiary); border-radius: 8px; transition: var(--transition);
        }
        .neuron-item:hover { background: var(--bg-hover); }
        .neuron-info { display: flex; align-items: center; gap: 12px; }
        .neuron-rank {
            width: 28px; height: 28px; background: var(--accent-blue); color: var(--bg-primary);
            border-radius: 50%; display: flex; align-items: center; justify-content: center;
            font-size: 12px; font-weight: 700;
        }
        .neuron-name { font-weight: 500; }
        .neuron-score { font-weight: 600; color: var(--accent-orange); }

        .interpretation-box {
            background: linear-gradient(135deg, rgba(79, 195, 247, 0.1) 0%, rgba(156, 39, 176, 0.1) 100%);
            border: 1px solid rgba(79, 195, 247, 0.3);
            border-radius: 12px; padding: 20px; margin-top: 16px;
        }
        .interpretation-title {
            font-size: 14px; font-weight: 600; color: var(--accent-blue);
            margin-bottom: 12px; display: flex; align-items: center; gap: 8px;
        }
        .interpretation-content { font-size: 14px; color: var(--text-secondary); line-height: 1.7; }
        .interpretation-content strong { color: var(--text-primary); }

        .compliance-box { padding: 16px 20px; border-radius: 12px; margin-top: 16px; }
        .compliance-pass { background: rgba(76, 175, 80, 0.1); border: 1px solid rgba(76, 175, 80, 0.3); }
        .compliance-fail { background: rgba(244, 67, 54, 0.1); border: 1px solid rgba(244, 67, 54, 0.3); }
        .compliance-header { display: flex; align-items: center; gap: 8px; font-weight: 600; margin-bottom: 8px; }
        .compliance-pass .compliance-header { color: var(--accent-green); }
        .compliance-fail .compliance-header { color: var(--accent-red); }
        .compliance-text { font-size: 13px; color: var(--text-secondary); }

        .export-toolbar { display: flex; gap: 8px; align-items: center; margin-top: 12px; }
        .export-btn {
            display: inline-flex; align-items: center; gap: 6px;
            padding: 8px 14px; border: 1px solid var(--border-color);
            border-radius: 8px; background: var(--bg-secondary);
            color: var(--text-secondary); font-size: 12px; font-weight: 500;
            cursor: pointer; transition: all 0.2s ease; font-family: inherit;
        }
        .export-btn:hover {
            background: var(--accent-blue); color: #fff;
            border-color: var(--accent-blue); transform: translateY(-1px);
        }
        .export-btn svg { flex-shrink: 0; }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    `;
}
