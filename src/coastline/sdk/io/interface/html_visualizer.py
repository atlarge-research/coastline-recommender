"""HTML visualization generator for recommendations."""

from datetime import datetime
from html import escape
from pathlib import Path
from typing import List, Union

from coastline.sdk.models.recommendation import Recommendation


def generate_html_report(
    recommendations: List[Recommendation], workload_info: dict, output_path: Union[str, Path]
) -> None:
    """Generate an interactive HTML report for the given recommendations."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPU Configuration Recommendations</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{ 
            max-width: 1200px; 
            margin: 0 auto; 
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; 
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{ font-size: 2em; margin-bottom: 10px; }}
        .header p {{ opacity: 0.9; }}
        .workload-info {{
            background: #f8f9fa;
            padding: 20px;
            border-bottom: 2px solid #e9ecef;
        }}
        .workload-info h2 {{ margin-bottom: 15px; color: #495057; }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .info-item {{
            background: white;
            padding: 12px;
            border-radius: 6px;
            border-left: 3px solid #667eea;
        }}
        .info-label {{ 
            font-size: 0.85em; 
            color: #6c757d; 
            margin-bottom: 4px;
        }}
        .info-value {{ 
            font-size: 1.1em; 
            font-weight: 600; 
            color: #212529;
        }}
        .recommendations {{ padding: 30px; }}
        .recommendations h2 {{ 
            margin-bottom: 20px; 
            color: #495057;
            font-size: 1.5em;
        }}
        .rec-card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 15px;
            border-left: 4px solid #667eea;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .rec-card:hover {{
            transform: translateX(5px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        .rec-card.best {{ border-left-color: #28a745; }}
        .rec-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .rec-title {{
            font-size: 1.3em;
            font-weight: 600;
            color: #212529;
        }}
        .rec-badge {{
            background: #28a745;
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        .rec-metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px;
        }}
        .metric {{
            background: white;
            padding: 12px;
            border-radius: 6px;
        }}
        .metric-label {{
            font-size: 0.8em;
            color: #6c757d;
            margin-bottom: 4px;
        }}
        .metric-value {{
            font-size: 1.2em;
            font-weight: 600;
            color: #212529;
        }}
        .metric-unit {{
            font-size: 0.85em;
            color: #6c757d;
            margin-left: 2px;
        }}
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #6c757d;
            font-size: 0.9em;
            border-top: 2px solid #e9ecef;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>GPU Configuration Recommendations</h1>
            <p>J→K→L→M Orchestration Engine</p>
            <p style="font-size: 0.9em; margin-top: 10px;">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
        
        <div class="workload-info">
            <h2>Workload Specification</h2>
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-label">Model</div>
                    <div class="info-value">{escape(str(workload_info.get("llm_model", "N/A")))}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Method</div>
                    <div class="info-value">{escape(str(workload_info.get("method", "N/A")).upper())}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">GPU Model</div>
                    <div class="info-value">{escape(str(workload_info.get("gpu_model", "N/A")))}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Tokens/Sample</div>
                    <div class="info-value">{workload_info.get("tokens_per_sample", "N/A")}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Batch Size</div>
                    <div class="info-value">{workload_info.get("batch_size", "N/A")}</div>
                </div>
            </div>
        </div>
        
        <div class="recommendations">
            <h2>Top Recommendations</h2>
"""

    for i, rec in enumerate(recommendations[:5], 1):
        is_best = i == 1
        card_class = "rec-card best" if is_best else "rec-card"

        power = rec.metadata.get("predicted_power_watts", 0)
        efficiency = rec.metadata.get("tokens_per_watt", 0)

        html += f"""
            <div class="{card_class}">
                <div class="rec-header">
                    <div class="rec-title">
                        #{i}: {rec.total_gpus} GPUs ({rec.gpus_per_node}×{rec.number_of_nodes})
                    </div>
                    {'<div class="rec-badge">BEST</div>' if is_best else ""}
                </div>
                
                <div class="rec-metrics">
                    <div class="metric">
                        <div class="metric-label">Throughput</div>
                        <div class="metric-value">
                            {rec.predicted_throughput:.1f}
                            <span class="metric-unit">tokens/sec</span>
                        </div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Power</div>
                        <div class="metric-value">
                            {power:.1f}
                            <span class="metric-unit">watts</span>
                        </div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Efficiency</div>
                        <div class="metric-value">
                            {efficiency:.2f}
                            <span class="metric-unit">tok/watt</span>
                        </div>
                    </div>
                </div>
            </div>
"""

    html += """
        </div>
        
        <div class="footer">
            <p>Generated by GPU Configuration Recommender System</p>
            <p style="margin-top: 5px;">Architecture: J→K→L→M Orchestration Loop</p>
        </div>
    </div>
</body>
</html>
"""

    with open(output_path, "w") as f:
        f.write(html)


def generate_comparison_chart(recommendations: List[Recommendation], output_path: Union[str, Path]) -> None:
    """Generate an HTML bar-chart comparing the top recommendations."""
    configs = [f"{r.total_gpus}G" for r in recommendations[:5]]
    throughputs = [r.predicted_throughput or 0 for r in recommendations[:5]]

    max_throughput = max(throughputs) if throughputs else 1

    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Performance Comparison</title>
    <style>
        body { 
            font-family: sans-serif; 
            padding: 40px;
            background: #f5f5f5;
        }
        .chart-container {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            max-width: 800px;
            margin: 0 auto;
        }
        h1 { text-align: center; color: #333; }
        .chart {
            display: flex;
            align-items: flex-end;
            height: 300px;
            margin: 40px 0;
            border-bottom: 2px solid #ddd;
            border-left: 2px solid #ddd;
            padding: 10px;
        }
        .bar-group {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 0 10px;
        }
        .bar {
            width: 100%;
            border-radius: 4px 4px 0 0;
            transition: all 0.3s;
        }
        .bar:hover { opacity: 0.8; }
        .bar.throughput { background: linear-gradient(180deg, #667eea, #764ba2); }
        .bar.power { background: linear-gradient(180deg, #f093fb, #f5576c); }
        .label {
            margin-top: 10px;
            font-weight: 600;
            color: #333;
        }
        .legend {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 20px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .legend-color {
            width: 20px;
            height: 20px;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="chart-container">
        <h1>Performance Comparison</h1>
        <div class="chart">
"""

    for config, throughput in zip(configs, throughputs):
        t_height = (throughput / max_throughput) * 250
        html += f"""
            <div class="bar-group">
                <div class="bar throughput" style="height: {t_height}px;"
                     title="Throughput: {throughput:.1f} tokens/sec"></div>
                <div class="label">{config}</div>
            </div>
"""

    html += """
        </div>
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background: linear-gradient(180deg, #667eea, #764ba2);"></div>
                <span>Throughput (tokens/sec)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: linear-gradient(180deg, #f093fb, #f5576c);"></div>
                <span>Power (watts)</span>
            </div>
        </div>
    </div>
</body>
</html>
"""

    with open(output_path, "w") as f:
        f.write(html)
