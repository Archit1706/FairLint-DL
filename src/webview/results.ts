import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { getWebviewHtml } from './htmlBuilder';
import { AnalysisResults } from './types';

export function showResults(results: AnalysisResults): void {
    const panel = vscode.window.createWebviewPanel(
        'fairnessResults',
        'Fairness Analysis Results',
        vscode.ViewColumn.One,
        { enableScripts: true, retainContextWhenHidden: true },
    );

    panel.webview.html = getWebviewHtml(results);

    panel.webview.onDidReceiveMessage(async (message) => {
        if (message.command === 'saveJson') {
            await handleSaveJson(results);
        }
    });
}

async function handleSaveJson(results: AnalysisResults): Promise<void> {
    const baseFileName = results.metadata?.file?.replace('.csv', '') || 'fairness-report';
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const defaultDir = results.metadata?.filePath
        ? path.dirname(results.metadata.filePath)
        : vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || '';
    const defaultName = `${baseFileName}_data_${timestamp}.json`;
    const defaultUri = defaultDir ? vscode.Uri.file(path.join(defaultDir, defaultName)) : undefined;

    const uri = await vscode.window.showSaveDialog({
        defaultUri,
        filters: { 'JSON Files': ['json'], 'All Files': ['*'] },
        title: 'Save Analysis Data (JSON)',
    });

    if (uri) {
        try {
            const exportData = {
                generated_at: new Date().toISOString(),
                metadata: results.metadata,
                training: {
                    accuracy: results.training.accuracy,
                    num_parameters: results.training.num_parameters,
                    protected_features: results.training.protected_features,
                    hidden_layers: results.training.hidden_layers,
                    dataset_info: results.training.dataset_info,
                    training_history: results.training.training_history,
                },
                qid_metrics: results.analysis.qid_metrics,
                search_results: {
                    best_qid: results.search.search_results.best_qid,
                    num_found: results.search.search_results.num_found,
                },
                layer_analysis: results.debug?.layer_analysis,
                neuron_analysis: results.debug?.neuron_analysis,
                explanations: {
                    shap_global_importance: results.explanations?.shap?.global_importance,
                    shap_feature_names: results.explanations?.shap?.feature_names,
                    lime_aggregated_importance: results.explanations?.lime?.aggregated_importance,
                    lime_feature_names: results.explanations?.lime?.feature_names,
                },
            };
            fs.writeFileSync(uri.fsPath, JSON.stringify(exportData, null, 2), 'utf-8');
            vscode.window.showInformationMessage(`Data exported to ${path.basename(uri.fsPath)}`);
        } catch (err: unknown) {
            const error = err as Error;
            vscode.window.showErrorMessage(`Failed to export data: ${error.message}`);
        }
    }
}
