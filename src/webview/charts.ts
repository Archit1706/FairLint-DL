// Generates the Plotly chart rendering JavaScript for the webview

import { SearchResults, QidMetrics, LayerAnalysis, NeuronAnalysis, ActivationsData, ShapData, LimeData } from './types';

export function getChartsScript(
    searchResults: SearchResults,
    qidMetrics: QidMetrics,
    layerAnalysis: LayerAnalysis | null,
    neuronAnalysis: NeuronAnalysis[] | null,
    activationsData: ActivationsData | null,
    explanationsData: { shap?: ShapData; lime?: LimeData } | null,
): string {
    return `
    <script>
        var chartLayout = {
            plot_bgcolor: '#252526',
            paper_bgcolor: '#252526',
            font: { color: '#e4e4e4', family: '-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif' },
            margin: { t: 20, r: 20, b: 50, l: 60 },
            xaxis: { gridcolor: '#3c3c3c', zerolinecolor: '#3c3c3c' },
            yaxis: { gridcolor: '#3c3c3c', zerolinecolor: '#3c3c3c' }
        };

        var instances = ${JSON.stringify(searchResults.discriminatory_instances)};
        var qidValues = instances.map(function(inst) { return inst.qid; });
        var disparateImpact = ${qidMetrics.mean_disparate_impact};

        // QID Histogram
        if (qidValues.length > 0) {
            Plotly.newPlot('qid-histogram', [{
                x: qidValues,
                type: 'histogram',
                nbinsx: 20,
                marker: { color: '#ff9800', line: { color: '#fff', width: 1 } },
                opacity: 0.85,
                hovertemplate: 'QID Range: %{x}<br>Count: %{y}<extra></extra>'
            }], {
                ...chartLayout,
                margin: { t: 10, r: 20, b: 50, l: 50 },
                xaxis: { ...chartLayout.xaxis, title: { text: 'QID (bits)', font: { color: '#a0a0a0' } } },
                yaxis: { ...chartLayout.yaxis, title: { text: 'Frequency', font: { color: '#a0a0a0' } } },
                bargap: 0.05
            }, { responsive: true });
        }

        // Disparate Impact Gauge
        var gaugeValue = disparateImpact * 100;
        var gaugeColor = disparateImpact >= 0.8 ? '#4caf50' : (disparateImpact >= 0.6 ? '#ff9800' : '#f44336');

        Plotly.newPlot('disparate-impact-gauge', [{
            type: 'indicator',
            mode: 'gauge+number+delta',
            value: gaugeValue,
            title: { text: 'Disparate Impact Ratio', font: { size: 16, color: '#e4e4e4' } },
            number: { suffix: '%', font: { size: 36, color: '#e4e4e4' } },
            delta: {
                reference: 80,
                increasing: { color: '#4caf50' },
                decreasing: { color: '#f44336' },
                font: { size: 14 }
            },
            gauge: {
                axis: { range: [0, 100], tickwidth: 1, tickcolor: '#3c3c3c', tickfont: { color: '#a0a0a0' } },
                bar: { color: gaugeColor, thickness: 0.75 },
                bgcolor: '#1e1e1e',
                borderwidth: 2,
                bordercolor: '#3c3c3c',
                steps: [
                    { range: [0, 60], color: 'rgba(244, 67, 54, 0.3)' },
                    { range: [60, 80], color: 'rgba(255, 152, 0, 0.3)' },
                    { range: [80, 100], color: 'rgba(76, 175, 80, 0.3)' }
                ],
                threshold: { line: { color: '#ff5252', width: 4 }, thickness: 0.75, value: 80 }
            }
        }], {
            ...chartLayout,
            margin: { t: 50, r: 30, b: 30, l: 30 }
        }, { responsive: true });

        // QID Scatter Plot
        if (instances.length > 0) {
            Plotly.newPlot('qid-chart', [{
                x: instances.map(function(_, i) { return i + 1; }),
                y: instances.map(function(inst) { return inst.qid; }),
                type: 'scatter',
                mode: 'markers',
                marker: {
                    size: 10,
                    color: instances.map(function(inst) { return inst.qid; }),
                    colorscale: [[0, '#4caf50'], [0.5, '#ff9800'], [1, '#f44336']],
                    showscale: true,
                    colorbar: { title: { text: 'QID (bits)', font: { color: '#e4e4e4' } }, tickfont: { color: '#a0a0a0' } }
                },
                text: instances.map(function(inst) { return 'QID: ' + inst.qid.toFixed(4) + ' bits<br>Variance: ' + inst.variance.toFixed(4); }),
                hoverinfo: 'text'
            }], {
                ...chartLayout,
                xaxis: { ...chartLayout.xaxis, title: { text: 'Instance Index', font: { color: '#a0a0a0' } } },
                yaxis: { ...chartLayout.yaxis, title: { text: 'QID (bits)', font: { color: '#a0a0a0' } } }
            }, { responsive: true });
        }

        // Layer Sensitivity Chart
        var layerData = ${JSON.stringify(layerAnalysis?.all_layers || [])};
        if (layerData.length > 0) {
            Plotly.newPlot('layer-chart', [{
                x: layerData.map(function(l) { return l.layer_name; }),
                y: layerData.map(function(l) { return l.sensitivity; }),
                type: 'bar',
                marker: {
                    color: layerData.map(function(l) { return l.sensitivity; }),
                    colorscale: [[0, '#4fc3f7'], [0.5, '#9c27b0'], [1, '#f44336']],
                    line: { width: 0 }
                },
                hovertemplate: '%{x}<br>Sensitivity: %{y:.4f}<extra></extra>'
            }], {
                ...chartLayout,
                xaxis: { ...chartLayout.xaxis, title: { text: 'Layer', font: { color: '#a0a0a0' } } },
                yaxis: { ...chartLayout.yaxis, title: { text: 'Sensitivity Score', font: { color: '#a0a0a0' } } }
            }, { responsive: true });
        }

        // Neuron Impact Chart
        var neuronData = ${JSON.stringify(neuronAnalysis || [])};
        if (neuronData.length > 0) {
            Plotly.newPlot('neuron-chart', [{
                x: neuronData.map(function(n) { return 'N' + n.neuron_idx; }),
                y: neuronData.map(function(n) { return n.impact_score; }),
                type: 'bar',
                marker: { color: '#f44336', line: { width: 0 } },
                hovertemplate: 'Neuron %{x}<br>Impact: %{y:.4f}<extra></extra>'
            }], {
                ...chartLayout,
                xaxis: { ...chartLayout.xaxis, title: { text: 'Neuron', font: { color: '#a0a0a0' } } },
                yaxis: { ...chartLayout.yaxis, title: { text: 'Impact Score', font: { color: '#a0a0a0' } } }
            }, { responsive: true });
        }

        // Internal Space Activation Charts
        var activationsData = ${JSON.stringify(activationsData)};
        if (activationsData && activationsData.layers) {
            activationsData.layers.forEach(function(layer, idx) {
                Plotly.newPlot('activation-layer-' + idx, [{
                    x: layer.x,
                    y: layer.y,
                    mode: 'markers',
                    type: 'scatter',
                    marker: {
                        size: 5,
                        color: activationsData.labels,
                        colorscale: [[0, '#4fc3f7'], [1, '#f44336']],
                        opacity: 0.6,
                        showscale: true,
                        colorbar: { title: { text: 'Label', font: { color: '#a0a0a0' } }, tickfont: { color: '#a0a0a0' } }
                    },
                    text: activationsData.labels.map(function(l, i) {
                        return 'Label: ' + l + '<br>Protected: ' + activationsData.protected[i].toFixed(2);
                    }),
                    hoverinfo: 'text'
                }], {
                    ...chartLayout,
                    xaxis: { ...chartLayout.xaxis, title: { text: 'Component 1', font: { color: '#a0a0a0' } } },
                    yaxis: { ...chartLayout.yaxis, title: { text: 'Component 2', font: { color: '#a0a0a0' } } }
                }, { responsive: true });
            });
        }

        // SHAP Charts
        var shapData = ${JSON.stringify(explanationsData?.shap || null)};
        if (shapData) {
            // Filter out NaN values from global importance
            var validImportance = shapData.global_importance.map(function(val) {
                return (val === null || isNaN(val)) ? 0 : val;
            });

            var shapIndices = validImportance
                .map(function(val, idx) { return { val: val, idx: idx }; })
                .sort(function(a, b) { return b.val - a.val; });

            // Global importance bar chart
            Plotly.newPlot('shap-global-chart', [{
                y: shapIndices.map(function(d) { return shapData.feature_names[d.idx]; }),
                x: shapIndices.map(function(d) { return d.val; }),
                type: 'bar',
                orientation: 'h',
                marker: {
                    color: shapIndices.map(function(d) { return d.val; }),
                    colorscale: [[0, '#4fc3f7'], [0.5, '#ff9800'], [1, '#f44336']],
                    line: { width: 0 }
                },
                hovertemplate: '%{y}: %{x:.4f}<extra></extra>'
            }], {
                ...chartLayout,
                margin: { t: 20, r: 20, b: 50, l: 150 },
                yaxis: { ...chartLayout.yaxis, autorange: 'reversed' },
                xaxis: { ...chartLayout.xaxis, title: { text: 'Mean |SHAP Value| (log-odds)', font: { color: '#a0a0a0' } } }
            }, { responsive: true });

            // Beeswarm chart
            if (shapData.shap_values && shapData.shap_values.length > 0) {
                var numFeatures = shapData.feature_names.length;
                var numInstances = shapData.shap_values.length;
                var hasFeatureValues = shapData.feature_values && shapData.feature_values.length > 0;

                var beeswarmTraces = [];
                var sortedFeatureIndices = shapIndices.map(function(d) { return d.idx; });

                for (var fi = 0; fi < sortedFeatureIndices.length; fi++) {
                    var featIdx = sortedFeatureIndices[fi];
                    var featName = shapData.feature_names[featIdx];
                    var shapVals = [];
                    var featVals = [];
                    var jitterY = [];

                    for (var inst = 0; inst < numInstances; inst++) {
                        var sv = shapData.shap_values[inst][featIdx];
                        if (sv !== null && !isNaN(sv)) {
                            shapVals.push(sv);
                            if (hasFeatureValues) {
                                featVals.push(shapData.feature_values[inst][featIdx]);
                            }
                            jitterY.push(fi + (Math.random() - 0.5) * 0.4);
                        }
                    }

                    if (shapVals.length > 0) {
                        var traceConfig = {
                            x: shapVals,
                            y: jitterY,
                            type: 'scatter',
                            mode: 'markers',
                            marker: {
                                size: 6,
                                opacity: 0.7,
                            },
                            name: featName,
                            showlegend: false,
                            hovertemplate: featName + '<br>SHAP value: %{x:.4f}<extra></extra>'
                        };

                        if (hasFeatureValues && featVals.length > 0) {
                            traceConfig.marker.color = featVals;
                            traceConfig.marker.colorscale = [[0, '#4fc3f7'], [1, '#f44336']];
                            traceConfig.marker.showscale = fi === 0;
                            if (fi === 0) {
                                traceConfig.marker.colorbar = {
                                    title: { text: 'Feature Value', font: { color: '#a0a0a0', size: 11 } },
                                    tickfont: { color: '#a0a0a0', size: 10 },
                                    thickness: 15,
                                    len: 0.5
                                };
                            }
                            traceConfig.hovertemplate = featName + '<br>SHAP value: %{x:.4f}<br>Feature value: %{marker.color:.2f}<extra></extra>';
                        } else {
                            traceConfig.marker.color = shapVals;
                            traceConfig.marker.colorscale = [[0, '#4fc3f7'], [0.5, '#a0a0a0'], [1, '#f44336']];
                            traceConfig.marker.showscale = fi === 0;
                            if (fi === 0) {
                                traceConfig.marker.colorbar = {
                                    title: { text: 'SHAP Value', font: { color: '#a0a0a0', size: 11 } },
                                    tickfont: { color: '#a0a0a0', size: 10 },
                                    thickness: 15,
                                    len: 0.5
                                };
                            }
                        }

                        beeswarmTraces.push(traceConfig);
                    }
                }

                var tickLabels = sortedFeatureIndices.map(function(idx) { return shapData.feature_names[idx]; });
                var tickPositions = sortedFeatureIndices.map(function(_, i) { return i; });

                Plotly.newPlot('shap-beeswarm-chart', beeswarmTraces, {
                    ...chartLayout,
                    margin: { t: 20, r: 80, b: 50, l: 150 },
                    showlegend: false,
                    xaxis: {
                        ...chartLayout.xaxis,
                        title: { text: 'SHAP Value (log-odds impact)', font: { color: '#a0a0a0' } },
                        zeroline: true,
                        zerolinecolor: '#6e6e6e',
                        zerolinewidth: 1
                    },
                    yaxis: {
                        ...chartLayout.yaxis,
                        tickvals: tickPositions,
                        ticktext: tickLabels,
                        autorange: false,
                        range: [-0.5, sortedFeatureIndices.length - 0.5]
                    }
                }, { responsive: true });
            }

            // SHAP Scatter Plot for top feature
            if (shapData.shap_values && shapData.shap_values.length > 0 && hasFeatureValues) {
                var topFeatIdx = shapIndices[0].idx;
                var topFeatName = shapData.feature_names[topFeatIdx];
                var scatterX = [];
                var scatterY = [];
                var scatterColors = [];

                for (var si = 0; si < numInstances; si++) {
                    var fv = shapData.feature_values[si][topFeatIdx];
                    var shv = shapData.shap_values[si][topFeatIdx];
                    if (fv !== null && !isNaN(fv) && shv !== null && !isNaN(shv)) {
                        scatterX.push(fv);
                        scatterY.push(shv);
                        // Color by another feature if available, otherwise by shap value
                        if (shapIndices.length > 1) {
                            var secondFeatIdx = shapIndices[1].idx;
                            scatterColors.push(shapData.feature_values[si][secondFeatIdx]);
                        } else {
                            scatterColors.push(shv);
                        }
                    }
                }

                var scatterColorbarTitle = shapIndices.length > 1
                    ? shapData.feature_names[shapIndices[1].idx]
                    : 'SHAP Value';

                Plotly.newPlot('shap-scatter-chart', [{
                    x: scatterX,
                    y: scatterY,
                    type: 'scatter',
                    mode: 'markers',
                    marker: {
                        size: 8,
                        color: scatterColors,
                        colorscale: [[0, '#4fc3f7'], [0.5, '#a0a0a0'], [1, '#f44336']],
                        showscale: true,
                        opacity: 0.7,
                        colorbar: {
                            title: { text: scatterColorbarTitle, font: { color: '#a0a0a0', size: 11 } },
                            tickfont: { color: '#a0a0a0', size: 10 },
                            thickness: 15
                        }
                    },
                    hovertemplate: topFeatName + ': %{x:.3f}<br>SHAP: %{y:.4f}<extra></extra>'
                }], {
                    ...chartLayout,
                    margin: { t: 10, r: 80, b: 50, l: 80 },
                    xaxis: { ...chartLayout.xaxis, title: { text: topFeatName + ' (feature value)', font: { color: '#a0a0a0' } } },
                    yaxis: { ...chartLayout.yaxis, title: { text: 'SHAP Value (log-odds)', font: { color: '#a0a0a0' } } }
                }, { responsive: true });
            }

            // SHAP Heatmap (instances x features)
            if (shapData.shap_values && shapData.shap_values.length > 0) {
                var heatmapZ = [];
                var heatmapY = [];
                var heatmapX = sortedFeatureIndices.map(function(idx) { return shapData.feature_names[idx]; });

                for (var hi = 0; hi < numInstances; hi++) {
                    var row = sortedFeatureIndices.map(function(idx) {
                        var v = shapData.shap_values[hi][idx];
                        return (v === null || isNaN(v)) ? 0 : v;
                    });
                    heatmapZ.push(row);
                    heatmapY.push('Instance ' + (hi + 1));
                }

                Plotly.newPlot('shap-heatmap-chart', [{
                    z: heatmapZ,
                    x: heatmapX,
                    y: heatmapY,
                    type: 'heatmap',
                    colorscale: [[0, '#4fc3f7'], [0.5, '#252526'], [1, '#f44336']],
                    zmid: 0,
                    colorbar: {
                        title: { text: 'SHAP Value', font: { color: '#a0a0a0', size: 11 } },
                        tickfont: { color: '#a0a0a0', size: 10 },
                        thickness: 15
                    },
                    hovertemplate: '%{x}<br>%{y}<br>SHAP: %{z:.4f}<extra></extra>'
                }], {
                    ...chartLayout,
                    margin: { t: 10, r: 80, b: 100, l: 80 },
                    xaxis: { ...chartLayout.xaxis, tickangle: -45 },
                    yaxis: { ...chartLayout.yaxis }
                }, { responsive: true });
            }
        }

        // LIME Charts
        var limeData = ${JSON.stringify(explanationsData?.lime || null)};
        if (limeData) {
            var validLimeImportance = limeData.aggregated_importance.map(function(val) {
                return (val === null || isNaN(val)) ? 0 : val;
            });

            var limeIndices = validLimeImportance
                .map(function(val, idx) { return { val: val, idx: idx }; })
                .sort(function(a, b) { return b.val - a.val; });

            Plotly.newPlot('lime-global-chart', [{
                y: limeIndices.map(function(d) { return limeData.feature_names[d.idx]; }),
                x: limeIndices.map(function(d) { return d.val; }),
                type: 'bar',
                orientation: 'h',
                marker: {
                    color: limeIndices.map(function(d) { return d.val; }),
                    colorscale: [[0, '#4fc3f7'], [0.5, '#ff9800'], [1, '#f44336']],
                    line: { width: 0 }
                },
                hovertemplate: '%{y}: %{x:.4f}<extra></extra>'
            }], {
                ...chartLayout,
                margin: { t: 20, r: 20, b: 50, l: 150 },
                yaxis: { ...chartLayout.yaxis, autorange: 'reversed' },
                xaxis: { ...chartLayout.xaxis, title: { text: 'Mean |LIME Weight|', font: { color: '#a0a0a0' } } }
            }, { responsive: true });
        }
    </script>`;
}
