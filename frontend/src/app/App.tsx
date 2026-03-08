import { useState, useCallback } from "react";
import { Header } from "./components/Header";
import { TrainingPanel } from "./components/TrainingPanel";
import { RandomForestVisualization } from "./components/RandomForestVisualization";
import { ControlsPanel } from "./components/ControlsPanel";
import {
  getMetrics,
  getFeatureImportance,
  getModelInfo,
  explainModel,
  MetricsResponse,
  ModelInfoResponse,
} from "./services/api";

export interface ModelState {
  isTrained: boolean;
  isTraining: boolean;
  dataType: "csv" | "image" | null;
  mode: "Classification" | "Regression" | null;
  metrics: MetricsResponse | null;
  featureImportance: [string, number][];
  features: string[];
  classes: string[];
  error: string | null;
  modelExplanation: string | null;
}

export default function App() {
  const [trainingMode, setTrainingMode] = useState<"tabular" | "image">(
    "tabular",
  );
  const [trees, setTrees] = useState(100);
  const [maxDepth, setMaxDepth] = useState(10);
  const [minSamplesSplit, setMinSamplesSplit] = useState(2);
  const [minSamplesLeaf, setMinSamplesLeaf] = useState(1);

  const [modelState, setModelState] = useState<ModelState>({
    isTrained: false,
    isTraining: false,
    dataType: null,
    mode: null,
    metrics: null,
    featureImportance: [],
    features: [],
    classes: [],
    error: null,
    modelExplanation: null,
  });

  const setTrainingStatus = useCallback(
    (isTraining: boolean, error?: string) => {
      setModelState((prev) => ({
        ...prev,
        isTraining,
        error: error || null,
      }));
    },
    [],
  );

  const onTrainingComplete = useCallback(async () => {
    try {
      const [metrics, featureImportance, modelInfo, explanationResult] =
        await Promise.all([
          getMetrics(),
          getFeatureImportance().catch(() => [] as [string, number][]),
          getModelInfo(),
          explainModel().catch(() => ({ explanation: null })),
        ]);

      setModelState((prev) => ({
        ...prev,
        isTrained: true,
        isTraining: false,
        dataType: modelInfo.data_type,
        mode: modelInfo.mode as "Classification" | "Regression",
        metrics,
        featureImportance,
        features: modelInfo.features || [],
        classes: modelInfo.classes || [],
        error: null,
        modelExplanation: explanationResult.explanation,
      }));
    } catch (error) {
      setModelState((prev) => ({
        ...prev,
        isTraining: false,
        error:
          error instanceof Error ? error.message : "Failed to fetch model data",
      }));
    }
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#0A0A0F] to-[#14141F] text-white">
      <Header />

      <div className="flex gap-6 p-6">
        {/* Left Panel - Training Input (30%) */}
        <div className="w-[30%]">
          <TrainingPanel
            mode={trainingMode}
            onModeChange={setTrainingMode}
            modelParams={{ trees, maxDepth, minSamplesSplit, minSamplesLeaf }}
            onTreesChange={setTrees}
            onMaxDepthChange={setMaxDepth}
            onMinSamplesSplitChange={setMinSamplesSplit}
            onMinSamplesLeafChange={setMinSamplesLeaf}
            isTraining={modelState.isTraining}
            onTrainingStart={() => setTrainingStatus(true)}
            onTrainingComplete={onTrainingComplete}
            onTrainingError={(err) => setTrainingStatus(false, err)}
          />
        </div>

        {/* Middle Panel - Model Visualization (45%) */}
        <div className="w-[45%]">
          <RandomForestVisualization
            trees={trees}
            maxDepth={maxDepth}
            minSamplesSplit={minSamplesSplit}
            minSamplesLeaf={minSamplesLeaf}
            modelState={modelState}
          />
        </div>

        {/* Right Panel - Controls & Metrics (25%) */}
        <div className="w-[25%]">
          <ControlsPanel
            trees={trees}
            maxDepth={maxDepth}
            minSamplesSplit={minSamplesSplit}
            minSamplesLeaf={minSamplesLeaf}
            trainingMode={trainingMode}
            modelState={modelState}
          />
        </div>
      </div>
    </div>
  );
}
