import * as vscode from 'vscode';
import * as path from 'path';
import axios from 'axios';
import { getConfig, getServerUrl } from '../config/settings';
import { updateStatusBar, setStatusBarSuccess, setStatusBarError } from '../server/statusBar';
import { parseError, showError } from '../server/errors';
import { showResults } from '../webview/results';

export async function runAnalysis(
    filePath: string,
    labelColumn: string,
    protectedFeatures: string[],
    hiddenLayers: number[],
): Promise<void> {
    const config = getConfig();
    const serverUrl = getServerUrl();

    await vscode.window.withProgress(
        {
            location: config.showProgress ? vscode.ProgressLocation.Notification : vscode.ProgressLocation.Window,
            title: 'Fairness Analysis',
            cancellable: false,
        },
        async (progress) => {
            const startTime = Date.now();
            const fileName = path.basename(filePath);

            try {
                // Step 1/6: Train model (0-30%)
                progress.report({ increment: 0, message: 'Step 1/6: Training neural network model...' });
                updateStatusBar('$(sync~spin) Training DNN...', `Training on ${fileName}`);

                const trainResponse = await axios.post(
                    `${serverUrl}/train`,
                    {
                        file_path: filePath,
                        label_column: labelColumn,
                        sensitive_features: protectedFeatures,
                        num_epochs: config.epochs,
                        batch_size: config.batchSize,
                        hidden_layers: hiddenLayers,
                    },
                    { timeout: 300000 },
                );

                const trainTime = Math.round((Date.now() - startTime) / 1000);
                progress.report({
                    increment: 30,
                    message: `Step 1/6: Training complete (${trainTime}s) - ${trainResponse.data.accuracy.toFixed(1)}% accuracy`,
                });
                updateStatusBar(
                    `$(check) Trained [${trainResponse.data.accuracy.toFixed(0)}%]`,
                    `Training complete: ${trainResponse.data.accuracy.toFixed(1)}% accuracy`,
                );

                // Step 2/6: Internal space visualization (30-38%)
                progress.report({ increment: 0, message: 'Step 2/6: Computing internal space visualization...' });
                updateStatusBar('$(eye) Computing activations...', 'Dimensionality reduction');

                const activationsResponse = await axios.post(
                    `${serverUrl}/activations`,
                    { method: 'pca', max_samples: config.maxSamples },
                    { timeout: 60000 },
                );

                progress.report({ increment: 8, message: 'Step 2/6: Internal space computed' });

                // Step 3/6: QID Analysis (38-55%)
                progress.report({ increment: 0, message: 'Step 3/6: Computing fairness metrics (QID analysis)...' });
                updateStatusBar('$(beaker) Computing QID...', 'Calculating fairness metrics');

                const protectedValues: Record<string, number[]> = {};
                protectedFeatures.forEach((_feature: string, idx: number) => {
                    protectedValues[idx.toString()] = [0.0, 1.0];
                });

                const analyzeResponse = await axios.post(
                    `${serverUrl}/analyze`,
                    {
                        file_path: filePath,
                        label_column: labelColumn,
                        sensitive_features: protectedFeatures,
                        protected_values: protectedValues,
                        max_samples: config.maxSamples,
                        qid_threshold: config.qidThreshold,
                    },
                    { timeout: 120000 },
                );

                progress.report({
                    increment: 17,
                    message: `Step 3/6: QID metrics computed - Mean QID: ${analyzeResponse.data.qid_metrics.mean_qid.toFixed(4)} bits`,
                });
                updateStatusBar(
                    `$(graph) QID: ${analyzeResponse.data.qid_metrics.mean_qid.toFixed(2)} bits`,
                    `Mean QID: ${analyzeResponse.data.qid_metrics.mean_qid.toFixed(4)} bits`,
                );

                // Step 4/6: Search (55-70%)
                progress.report({ increment: 0, message: 'Step 4/6: Searching for discriminatory instances...' });
                updateStatusBar('$(search) Searching...', 'Finding discriminatory instances');

                const searchResponse = await axios.post(
                    `${serverUrl}/search`,
                    {
                        protected_values: protectedValues,
                        num_iterations: config.globalIterations,
                        num_neighbors: config.localNeighbors,
                    },
                    { timeout: 120000 },
                );

                progress.report({
                    increment: 15,
                    message: `Step 4/6: Found ${searchResponse.data.search_results.num_found} discriminatory instances`,
                });
                updateStatusBar(
                    `$(bug) Found ${searchResponse.data.search_results.num_found}`,
                    `Found ${searchResponse.data.search_results.num_found} discriminatory instances`,
                );

                // Step 5/6: Debug (70-82%)
                progress.report({ increment: 0, message: 'Step 5/6: Localizing biased layers and neurons...' });
                updateStatusBar('$(telescope) Debugging...', 'Localizing biased neurons');

                const debugResponse = await axios.post(
                    `${serverUrl}/debug`,
                    {
                        protected_values: protectedValues,
                        num_iterations: config.globalIterations,
                        num_neighbors: config.localNeighbors,
                    },
                    { timeout: 120000 },
                );

                progress.report({ increment: 12, message: 'Step 5/6: Causal debugging complete' });

                // Step 6/6: LIME & SHAP Explanations (82-100%)
                progress.report({ increment: 0, message: 'Step 6/6: Computing LIME & SHAP explanations...' });
                updateStatusBar('$(lightbulb) Computing explanations...', 'Running LIME and SHAP');

                const explainResponse = await axios.post(
                    `${serverUrl}/explain`,
                    { method: 'both', num_instances: 10, max_background: 100 },
                    { timeout: 180000 },
                );

                const totalTime = Math.round((Date.now() - startTime) / 1000);
                progress.report({ increment: 18, message: `Analysis complete! Total time: ${totalTime}s` });

                await new Promise((resolve) => setTimeout(resolve, 500));

                showResults({
                    training: trainResponse.data,
                    analysis: analyzeResponse.data,
                    search: searchResponse.data,
                    debug: debugResponse.data,
                    activations: activationsResponse.data.activations,
                    explanations: explainResponse.data.explanations,
                    metadata: {
                        file: fileName,
                        filePath: filePath,
                        labelColumn: labelColumn,
                        totalTime: totalTime,
                    },
                });

                setStatusBarSuccess(`Done [${analyzeResponse.data.qid_metrics.num_discriminatory} issues]`);

                vscode.window.showInformationMessage(
                    `Fairness analysis complete! Found ${analyzeResponse.data.qid_metrics.num_discriminatory} potentially discriminatory instances.`,
                );
            } catch (error: unknown) {
                console.error('Analysis error:', error);
                const parsed = parseError(error);
                showError(parsed.title, `${parsed.detail}\n\nSuggestion: ${parsed.suggestion}`);
                setStatusBarError('Analysis failed');
            }
        },
    );
}
