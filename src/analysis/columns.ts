import axios from 'axios';
import { getServerUrl } from '../config/settings';
import { parseError, showError } from '../server/errors';

export interface ColumnData {
    columns: string[];
    sampleData: Record<string, unknown>[];
    detectedSensitive: string[];
}

export async function fetchColumns(filePath: string): Promise<ColumnData | null> {
    const serverUrl = getServerUrl();
    try {
        const response = await axios.post(`${serverUrl}/columns`, { file_path: filePath }, { timeout: 10000 });

        return {
            columns: response.data.columns,
            sampleData: response.data.sample_data,
            detectedSensitive: response.data.detected_sensitive || [],
        };
    } catch (error) {
        const parsed = parseError(error);
        showError(parsed.title, `${parsed.detail}\n${parsed.suggestion}`);
        return null;
    }
}
