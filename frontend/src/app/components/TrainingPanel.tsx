import { useState, useRef } from "react";
import {
  Upload,
  Check,
  Plus,
  FileText,
  X,
  Loader2,
  ChevronDown,
} from "lucide-react";
import { trainCSV, trainImages, parseCSVHeader } from "../services/api";

type TrainingMode = "tabular" | "image";
type ModelMode = "Classification" | "Regression";

interface TrainingPanelProps {
  mode: TrainingMode;
  onModeChange: (mode: TrainingMode) => void;
  modelParams: {
    trees: number;
    maxDepth: number;
    minSamplesSplit: number;
    minSamplesLeaf: number;
  };
  onTreesChange: (value: number) => void;
  onMaxDepthChange: (value: number) => void;
  onMinSamplesSplitChange: (value: number) => void;
  onMinSamplesLeafChange: (value: number) => void;
  isTraining: boolean;
  onTrainingStart: () => void;
  onTrainingComplete: () => void;
  onTrainingError: (error: string) => void;
}

export function TrainingPanel({
  mode,
  onModeChange,
  modelParams,
  onTreesChange,
  onMaxDepthChange,
  onMinSamplesSplitChange,
  onMinSamplesLeafChange,
  isTraining,
  onTrainingStart,
  onTrainingComplete,
  onTrainingError,
}: TrainingPanelProps) {
  // CSV state
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([]);
  const [targetColumn, setTargetColumn] = useState<string>("");
  const [modelMode, setModelMode] = useState<ModelMode>("Classification");
  const [featuresExpanded, setFeaturesExpanded] = useState(true);

  // Image state
  const [zipFiles, setZipFiles] = useState<File[]>([]);

  const csvInputRef = useRef<HTMLInputElement>(null);
  const zipInputRef = useRef<HTMLInputElement>(null);

  const handleCSVUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      const cols = await parseCSVHeader(file);
      setCsvFile(file);
      setColumns(cols);
      setSelectedFeatures(cols.slice(0, -1)); // Select all except last by default
      setTargetColumn(cols[cols.length - 1]); // Last column as target by default
    } catch (error) {
      onTrainingError("Failed to parse CSV file");
    }
  };

  const handleZipUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    const newFiles = Array.from(files).filter((f) => f.name.endsWith(".zip"));
    setZipFiles((prev) => [...prev, ...newFiles]);
  };

  const removeZipFile = (index: number) => {
    setZipFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const toggleFeature = (feature: string) => {
    if (feature === targetColumn) return; // Can't select target as feature
    setSelectedFeatures((prev) =>
      prev.includes(feature)
        ? prev.filter((f) => f !== feature)
        : [...prev, feature],
    );
  };

  const handleTargetChange = (newTarget: string) => {
    setTargetColumn(newTarget);
    // Remove new target from features if it was selected
    setSelectedFeatures((prev) => prev.filter((f) => f !== newTarget));
  };

  const handleTrainCSV = async () => {
    if (!csvFile || selectedFeatures.length === 0 || !targetColumn) {
      onTrainingError("Please select a file, features, and target column");
      return;
    }

    onTrainingStart();

    try {
      await trainCSV({
        file: csvFile,
        mode: modelMode,
        target: targetColumn,
        features: selectedFeatures,
        n_estimators: modelParams.trees,
        max_depth: modelParams.maxDepth,
        min_samples_split: modelParams.minSamplesSplit,
        min_samples_leaf: modelParams.minSamplesLeaf,
      });
      onTrainingComplete();
    } catch (error) {
      onTrainingError(
        error instanceof Error ? error.message : "Training failed",
      );
    }
  };

  const handleTrainImages = async () => {
    if (zipFiles.length < 2) {
      onTrainingError("Please upload at least 2 zip files (one per class)");
      return;
    }

    onTrainingStart();

    try {
      await trainImages({
        zipFiles,
        n_estimators: modelParams.trees,
        max_depth: modelParams.maxDepth,
        min_samples_split: modelParams.minSamplesSplit,
        min_samples_leaf: modelParams.minSamplesLeaf,
      });
      onTrainingComplete();
    } catch (error) {
      onTrainingError(
        error instanceof Error ? error.message : "Training failed",
      );
    }
  };

  return (
    <div className="flex flex-col gap-4">
      {/* Toggle Tabs */}
      <div className="flex gap-2 p-1 rounded-lg bg-white/5 backdrop-blur-sm border border-white/10">
        <button
          onClick={() => onModeChange("tabular")}
          disabled={isTraining}
          className={`flex-1 px-4 py-2 rounded-md text-sm font-medium transition-all ${
            mode === "tabular"
              ? "bg-[#39FF14]/20 text-[#39FF14] shadow-[0_0_20px_rgba(57,255,20,0.3)]"
              : "text-gray-400 hover:text-white"
          } disabled:opacity-50`}
        >
          Tabular
        </button>
        <button
          onClick={() => onModeChange("image")}
          disabled={isTraining}
          className={`flex-1 px-4 py-2 rounded-md text-sm font-medium transition-all ${
            mode === "image"
              ? "bg-[#00F0FF]/20 text-[#00F0FF] shadow-[0_0_20px_rgba(0,240,255,0.3)]"
              : "text-gray-400 hover:text-white"
          } disabled:opacity-50`}
        >
          Image
        </button>
      </div>

      <div className="space-y-4">
        {mode === "tabular" ? (
          <>
            {/* File Upload */}
            <div className="p-4 rounded-xl bg-white/5 backdrop-blur-sm border border-white/10">
              <input
                type="file"
                ref={csvInputRef}
                accept=".csv"
                onChange={handleCSVUpload}
                className="hidden"
              />
              <button
                onClick={() => csvInputRef.current?.click()}
                disabled={isTraining}
                className="w-full px-4 py-3 rounded-lg bg-white/10 hover:bg-white/15 border border-white/20 text-white font-medium flex items-center justify-center gap-2 transition-all disabled:opacity-50"
              >
                <Upload className="w-5 h-5" />
                {csvFile ? "Change File" : "Choose CSV File"}
              </button>
              {csvFile && (
                <div className="mt-3 flex items-center gap-2 text-sm text-gray-300">
                  <FileText className="w-4 h-4 text-[#39FF14]" />
                  <span className="truncate">{csvFile.name}</span>
                  <Check className="w-4 h-4 text-[#39FF14] ml-auto" />
                </div>
              )}
            </div>

            {/* Features Selection */}
            {columns.length > 0 && (
              <div className="p-4 rounded-xl bg-white/5 backdrop-blur-sm border border-white/10">
                <button
                  onClick={() => setFeaturesExpanded(!featuresExpanded)}
                  className="w-full flex items-center justify-between text-sm font-semibold text-white mb-3"
                >
                  <span>
                    Remove Leakage Columns ({selectedFeatures.length} selected)
                  </span>
                  <ChevronDown
                    className={`w-4 h-4 transition-transform ${featuresExpanded ? "rotate-180" : ""}`}
                  />
                </button>
                {featuresExpanded && (
                  <div className="space-y-2">
                    {columns
                      .filter((col) => col !== targetColumn)
                      .map((column) => (
                        <label
                          key={column}
                          className="flex items-center gap-3 p-2 rounded-lg hover:bg-white/5 cursor-pointer transition-colors"
                        >
                          <input
                            type="checkbox"
                            checked={selectedFeatures.includes(column)}
                            onChange={() => toggleFeature(column)}
                            disabled={isTraining}
                            className="w-4 h-4 rounded border-gray-600 bg-transparent checked:bg-[#39FF14] checked:border-[#39FF14] focus:ring-[#39FF14] focus:ring-offset-0"
                          />
                          <span className="text-sm text-gray-300">
                            {column}
                          </span>
                        </label>
                      ))}
                  </div>
                )}
              </div>
            )}

            {/* Target Column */}
            {columns.length > 0 && (
              <div className="p-4 rounded-xl bg-white/5 backdrop-blur-sm border border-white/10">
                <h3 className="text-sm font-semibold text-white mb-3">
                  Target Column
                </h3>
                <select
                  value={targetColumn}
                  onChange={(e) => handleTargetChange(e.target.value)}
                  disabled={isTraining}
                  className="w-full px-4 py-2 rounded-lg bg-white/10 border border-white/20 text-white focus:outline-none focus:ring-2 focus:ring-[#39FF14] disabled:opacity-50"
                >
                  {columns.map((column) => (
                    <option
                      className="text-black bg-white"
                      key={column}
                      value={column}
                    >
                      {column}
                    </option>
                  ))}
                </select>
              </div>
            )}

            {/* Mode Selector */}
            <div className="flex gap-2">
              <button
                onClick={() => setModelMode("Classification")}
                disabled={isTraining}
                className={`flex-1 px-4 py-2 rounded-lg font-medium text-sm transition-all ${
                  modelMode === "Classification"
                    ? "bg-[#39FF14]/20 text-[#39FF14] border border-[#39FF14]/50 shadow-[0_0_15px_rgba(57,255,20,0.2)]"
                    : "bg-white/5 text-gray-400 border border-white/10 hover:bg-white/10"
                } disabled:opacity-50`}
              >
                Classification
              </button>
              <button
                onClick={() => setModelMode("Regression")}
                disabled={isTraining}
                className={`flex-1 px-4 py-2 rounded-lg font-medium text-sm transition-all ${
                  modelMode === "Regression"
                    ? "bg-[#39FF14]/20 text-[#39FF14] border border-[#39FF14]/50 shadow-[0_0_15px_rgba(57,255,20,0.2)]"
                    : "bg-white/5 text-gray-400 border border-white/10 hover:bg-white/10"
                } disabled:opacity-50`}
              >
                Regression
              </button>
            </div>

            {/* Parameters Card */}
            <div className="p-4 rounded-xl bg-white/5 backdrop-blur-sm border border-white/10">
              <h3 className="text-xs font-semibold text-gray-400 mb-4 uppercase">
                Parameters
              </h3>

              <div className="space-y-4">
                {/* Trees Slider */}
                <div>
                  <div className="flex justify-between mb-2">
                    <label className="text-sm text-gray-300">Trees</label>
                    <span className="text-sm text-white font-semibold">
                      {modelParams.trees}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="10"
                    max="500"
                    value={modelParams.trees}
                    onChange={(e) => onTreesChange(Number(e.target.value))}
                    className="w-full h-2 rounded-full appearance-none bg-white/10 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-[#39FF14] [&::-webkit-slider-thumb]:shadow-[0_0_10px_rgba(57,255,20,0.8)] [&::-webkit-slider-thumb]:cursor-pointer"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>10</span>
                    <span>500</span>
                  </div>
                </div>

                {/* Max Depth Slider */}
                <div>
                  <div className="flex justify-between mb-2">
                    <label className="text-sm text-gray-300">Max depth</label>
                    <span className="text-sm text-white font-semibold">
                      {modelParams.maxDepth}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="3"
                    max="30"
                    value={modelParams.maxDepth}
                    onChange={(e) => onMaxDepthChange(Number(e.target.value))}
                    className="w-full h-2 rounded-full appearance-none bg-white/10 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-[#39FF14] [&::-webkit-slider-thumb]:shadow-[0_0_10px_rgba(57,255,20,0.8)] [&::-webkit-slider-thumb]:cursor-pointer"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>3</span>
                    <span>30</span>
                  </div>
                </div>

                {/* Min Samples Split Slider */}
                <div>
                  <div className="flex justify-between mb-2">
                    <label className="text-sm text-gray-300">
                      Min samples split
                    </label>
                    <span className="text-sm text-white font-semibold">
                      {modelParams.minSamplesSplit}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="2"
                    max="20"
                    value={modelParams.minSamplesSplit}
                    onChange={(e) =>
                      onMinSamplesSplitChange(Number(e.target.value))
                    }
                    className="w-full h-2 rounded-full appearance-none bg-white/10 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-[#39FF14] [&::-webkit-slider-thumb]:shadow-[0_0_10px_rgba(57,255,20,0.8)] [&::-webkit-slider-thumb]:cursor-pointer"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>2</span>
                    <span>20</span>
                  </div>
                </div>

                {/* Min Samples Leaf Slider */}
                <div>
                  <div className="flex justify-between mb-2">
                    <label className="text-sm text-gray-300">
                      Min samples leaf
                    </label>
                    <span className="text-sm text-white font-semibold">
                      {modelParams.minSamplesLeaf}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="10"
                    value={modelParams.minSamplesLeaf}
                    onChange={(e) =>
                      onMinSamplesLeafChange(Number(e.target.value))
                    }
                    className="w-full h-2 rounded-full appearance-none bg-white/10 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-[#39FF14] [&::-webkit-slider-thumb]:shadow-[0_0_10px_rgba(57,255,20,0.8)] [&::-webkit-slider-thumb]:cursor-pointer"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>1</span>
                    <span>10</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Train Button */}
            <button
              onClick={handleTrainCSV}
              disabled={isTraining || !csvFile || selectedFeatures.length === 0}
              className="w-full px-6 py-4 rounded-xl bg-[#39FF14] hover:bg-[#39FF14]/90 text-black font-bold text-lg shadow-[0_0_30px_rgba(57,255,20,0.5)] hover:shadow-[0_0_40px_rgba(57,255,20,0.7)] transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isTraining ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  TRAINING...
                </>
              ) : (
                "TRAIN RANDOM FOREST"
              )}
            </button>
          </>
        ) : (
          <>
            {/* Image Upload */}
            <div className="p-4 rounded-xl bg-white/5 backdrop-blur-sm border border-white/10">
              <p className="text-sm text-gray-300 mb-3">
                Upload ZIP files (one per class)
              </p>

              <input
                type="file"
                ref={zipInputRef}
                accept=".zip"
                multiple
                onChange={handleZipUpload}
                className="hidden"
              />

              {zipFiles.length > 0 && (
                <div className="space-y-2 mb-3">
                  {zipFiles.map((file, index) => (
                    <div
                      key={index}
                      className="flex items-center gap-3 p-3 rounded-lg bg-white/10 border border-white/20"
                    >
                      <FileText className="w-5 h-5 text-[#00F0FF]" />
                      <span className="flex-1 text-sm text-white truncate">
                        {file.name}
                      </span>
                      <span className="text-xs text-gray-400">
                        Class: {file.name.replace(".zip", "")}
                      </span>
                      <button
                        onClick={() => removeZipFile(index)}
                        disabled={isTraining}
                        className="p-1 hover:bg-white/10 rounded transition-colors disabled:opacity-50"
                      >
                        <X className="w-4 h-4 text-gray-400 hover:text-red-400" />
                      </button>
                    </div>
                  ))}
                </div>
              )}

              <button
                onClick={() => zipInputRef.current?.click()}
                disabled={isTraining}
                className="w-full px-4 py-2 rounded-lg bg-white/10 hover:bg-white/15 border border-white/20 text-white font-medium flex items-center justify-center gap-2 transition-all disabled:opacity-50"
              >
                <Plus className="w-5 h-5" />
                Add Class ZIP
              </button>

              <p className="mt-3 text-xs text-gray-400 italic">
                Classes extracted from zip filenames
              </p>
            </div>

            {/* Parameters Card */}
            <div className="p-4 rounded-xl bg-white/5 backdrop-blur-sm border border-white/10">
              <h3 className="text-xs font-semibold text-gray-400 mb-4 uppercase">
                Parameters
              </h3>

              <div className="space-y-4">
                {/* Trees Slider */}
                <div>
                  <div className="flex justify-between mb-2">
                    <label className="text-sm text-gray-300">Trees</label>
                    <span className="text-sm text-white font-semibold">
                      {modelParams.trees}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="10"
                    max="500"
                    value={modelParams.trees}
                    onChange={(e) => onTreesChange(Number(e.target.value))}
                    className="w-full h-2 rounded-full appearance-none bg-white/10 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-[#00F0FF] [&::-webkit-slider-thumb]:shadow-[0_0_10px_rgba(0,240,255,0.8)] [&::-webkit-slider-thumb]:cursor-pointer"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>10</span>
                    <span>500</span>
                  </div>
                </div>

                {/* Max Depth Slider */}
                <div>
                  <div className="flex justify-between mb-2">
                    <label className="text-sm text-gray-300">Max depth</label>
                    <span className="text-sm text-white font-semibold">
                      {modelParams.maxDepth}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="3"
                    max="30"
                    value={modelParams.maxDepth}
                    onChange={(e) => onMaxDepthChange(Number(e.target.value))}
                    className="w-full h-2 rounded-full appearance-none bg-white/10 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-[#00F0FF] [&::-webkit-slider-thumb]:shadow-[0_0_10px_rgba(0,240,255,0.8)] [&::-webkit-slider-thumb]:cursor-pointer"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>3</span>
                    <span>30</span>
                  </div>
                </div>

                {/* Min Samples Split Slider */}
                <div>
                  <div className="flex justify-between mb-2">
                    <label className="text-sm text-gray-300">
                      Min samples split
                    </label>
                    <span className="text-sm text-white font-semibold">
                      {modelParams.minSamplesSplit}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="2"
                    max="20"
                    value={modelParams.minSamplesSplit}
                    onChange={(e) =>
                      onMinSamplesSplitChange(Number(e.target.value))
                    }
                    className="w-full h-2 rounded-full appearance-none bg-white/10 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-[#00F0FF] [&::-webkit-slider-thumb]:shadow-[0_0_10px_rgba(0,240,255,0.8)] [&::-webkit-slider-thumb]:cursor-pointer"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>2</span>
                    <span>20</span>
                  </div>
                </div>

                {/* Min Samples Leaf Slider */}
                <div>
                  <div className="flex justify-between mb-2">
                    <label className="text-sm text-gray-300">
                      Min samples leaf
                    </label>
                    <span className="text-sm text-white font-semibold">
                      {modelParams.minSamplesLeaf}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="10"
                    value={modelParams.minSamplesLeaf}
                    onChange={(e) =>
                      onMinSamplesLeafChange(Number(e.target.value))
                    }
                    className="w-full h-2 rounded-full appearance-none bg-white/10 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-[#00F0FF] [&::-webkit-slider-thumb]:shadow-[0_0_10px_rgba(0,240,255,0.8)] [&::-webkit-slider-thumb]:cursor-pointer"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>1</span>
                    <span>10</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Train Button */}
            <button
              onClick={handleTrainImages}
              disabled={isTraining || zipFiles.length < 2}
              className="w-full px-6 py-4 rounded-xl bg-[#00F0FF] hover:bg-[#00F0FF]/90 text-black font-bold text-lg shadow-[0_0_30px_rgba(0,240,255,0.5)] hover:shadow-[0_0_40px_rgba(0,240,255,0.7)] transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isTraining ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  TRAINING...
                </>
              ) : (
                "TRAIN IMAGE CLASSIFIER"
              )}
            </button>

            <p className="text-xs text-gray-400 text-center">
              Using MobileNetV2 feature extractor
            </p>
          </>
        )}
      </div>
    </div>
  );
}
