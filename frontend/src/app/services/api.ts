const API_BASE = 'http://localhost:5000';

export interface TrainCSVParams {
  file: File;
  mode: 'Classification' | 'Regression';
  target: string;
  features: string[];
  n_estimators: number;
  max_depth: number;
  min_samples_split: number;
  min_samples_leaf?: number;
}

export interface TrainCSVResponse {
  status: string;
  metrics: {
    accuracy?: number;
    r2_score?: number;
    mse?: number;
    samples?: number;
  };
  features: string[];
}

export interface TrainImagesParams {
  zipFiles: File[];
  n_estimators: number;
  max_depth: number;
  min_samples_split: number;
  min_samples_leaf?: number;
}

export interface TrainImagesResponse {
  status: string;
  classes: string[];
}

export interface PredictCSVParams {
  inputs: Record<string, string | number>;
}

export interface PredictCSVResponse {
  prediction: string;
  probabilities?: number[];
}

export interface PredictImageResponse {
  prediction: string;
}

export interface MetricsResponse {
  accuracy?: number;
  r2_score?: number;
  mse?: number;
  samples?: number;
}

export interface FeatureImportanceResponse {
  [key: string]: number;
}

export interface ModelInfoResponse {
  data_type: 'csv' | 'image';
  mode: string;
  features?: string[];
  n_features_after_encoding?: number;
  classes?: string[];
  feature_extractor?: string;
}

// Train CSV Model
export async function trainCSV(params: TrainCSVParams): Promise<TrainCSVResponse> {
  const formData = new FormData();
  formData.append('file', params.file);
  formData.append('mode', params.mode);
  formData.append('target', params.target);
  formData.append('features', params.features.join(','));
  formData.append('n_estimators', params.n_estimators.toString());
  formData.append('max_depth', params.max_depth.toString());
  formData.append('min_samples_split', params.min_samples_split.toString());
  if (params.min_samples_leaf) {
    formData.append('min_samples_leaf', params.min_samples_leaf.toString());
  }

  const response = await fetch(`${API_BASE}/train_csv`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to train CSV model');
  }

  return response.json();
}

// Train Image Model
export async function trainImages(params: TrainImagesParams): Promise<TrainImagesResponse> {
  const formData = new FormData();
  params.zipFiles.forEach(file => {
    formData.append('zip_files', file);
  });
  formData.append('n_estimators', params.n_estimators.toString());
  formData.append('max_depth', params.max_depth.toString());
  formData.append('min_samples_split', params.min_samples_split.toString());
  if (params.min_samples_leaf) {
    formData.append('min_samples_leaf', params.min_samples_leaf.toString());
  }

  const response = await fetch(`${API_BASE}/train_images`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to train image model');
  }

  return response.json();
}

// Predict CSV
export async function predictCSV(inputs: Record<string, string | number>): Promise<PredictCSVResponse> {
  const response = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ inputs }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Prediction failed');
  }

  return response.json();
}

// Predict Image
export async function predictImage(imageFile: File): Promise<PredictImageResponse> {
  const formData = new FormData();
  formData.append('image', imageFile);

  const response = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Prediction failed');
  }

  return response.json();
}

// Get Metrics
export async function getMetrics(): Promise<MetricsResponse> {
  const response = await fetch(`${API_BASE}/metrics`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to get metrics');
  }

  return response.json();
}

// Get Feature Importance
export async function getFeatureImportance(): Promise<[string, number][]> {
  const response = await fetch(`${API_BASE}/feature_importance`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to get feature importance');
  }

  return response.json();
}

// Get Model Info
export async function getModelInfo(): Promise<ModelInfoResponse> {
  const response = await fetch(`${API_BASE}/model_info`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to get model info');
  }

  return response.json();
}

// Parse CSV to get columns
export function parseCSVHeader(file: File): Promise<string[]> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result as string;
      const firstLine = text.split('\n')[0];
      const columns = firstLine.split(',').map(col => col.trim().replace(/"/g, ''));
      resolve(columns);
    };
    reader.onerror = () => reject(new Error('Failed to read CSV file'));
    reader.readAsText(file);
  });
}

// Get model explanation from Gemini
export async function explainModel(): Promise<{ explanation: string }> {
  const response = await fetch(`${API_BASE}/explain_model`);
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to explain model');
  }
  return response.json();
}

// Start GitHub OAuth flow (opens in new tab)
export function startGitHubAuth(prompt: string): void {
  const encodedPrompt = encodeURIComponent(prompt);
  window.open(`${API_BASE}/auth/github?prompt=${encodedPrompt}`, '_blank');
}


