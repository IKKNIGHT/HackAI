import { useState, useCallback, useRef, useEffect } from "react";
import { Header } from "./components/Header";
import { TrainingPanel } from "./components/TrainingPanel";
import { RandomForestVisualization } from "./components/RandomForestVisualization";
import { ControlsPanel } from "./components/ControlsPanel";
import { ChevronDown, Maximize, Minimize, Settings, TreePine, BarChart3, GripVertical } from "lucide-react";
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

  // Panel states - resizable panels
  const [panelWidths, setPanelWidths] = useState({
    training: 30,
    visualization: 45,
    controls: 25,
  });

  const [panelHeights, setPanelHeights] = useState({
    training: 100,
    visualization: 100,
    controls: 100,
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

  const togglePanel = (panelName: keyof typeof panels) => {
    setPanels(prev => ({
      ...prev,
      [panelName]: {
        ...prev[panelName],
        collapsed: !prev[panelName].collapsed,
        expanded: false
      }
    }));
  };

  const expandPanel = (panelName: keyof typeof panels) => {
    setPanels(prev => {
      const newPanels = { ...prev };
      // Reset all panels
      Object.keys(newPanels).forEach(key => {
        newPanels[key as keyof typeof panels] = {
          collapsed: false,
          expanded: false
        };
      });
      // Expand the selected panel
      newPanels[panelName] = {
        collapsed: false,
        expanded: true
      };
      return newPanels;
    });
  };

  const resetPanels = () => {
    setPanelWidths({
      training: 30,
      visualization: 45,
      controls: 25,
    });
    setPanelHeights({
      training: 100,
      visualization: 100,
      controls: 100,
    });
  };

  // Resizable panel drag handling
  const dragRefs = useRef<{ [key: string]: HTMLDivElement | null }>({});
  const isDraggingRef = useRef<string | null>(null);
  const startXRef = useRef<number>(0);
  const startYRef = useRef<number>(0);
  const startWidthsRef = useRef<{ [key: string]: number }>({});
  const startHeightsRef = useRef<{ [key: string]: number }>({});

  // Mouse event handlers for horizontal resizing
  const handleMouseDown = useCallback((e: React.MouseEvent, panelKey: string) => {
    e.preventDefault();
    isDraggingRef.current = panelKey;
    startXRef.current = e.clientX;
    startWidthsRef.current = { ...panelWidths };
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  }, [panelWidths]);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isDraggingRef.current || isDraggingRef.current === 'vertical') return;

    const panelKey = isDraggingRef.current;
    const deltaX = e.clientX - startXRef.current;
    const containerWidth = window.innerWidth - 96; // Account for padding
    const deltaPercent = (deltaX / containerWidth) * 100;

    setPanelWidths(prev => {
      const newWidths = { ...startWidthsRef.current };

      if (panelKey === 'training') {
        const newWidth = Math.max(15, Math.min(70, startWidthsRef.current.training + deltaPercent));
        const remaining = 100 - newWidth;
        const totalOther = startWidthsRef.current.visualization + startWidthsRef.current.controls;
        const visRatio = startWidthsRef.current.visualization / totalOther;
        newWidths.training = newWidth;
        newWidths.visualization = Math.max(15, remaining * visRatio);
        newWidths.controls = Math.max(15, remaining * (1 - visRatio));
      } else if (panelKey === 'visualization') {
        const newWidth = Math.max(20, Math.min(80, startWidthsRef.current.visualization + deltaPercent));
        const remaining = 100 - startWidthsRef.current.training - newWidth;
        newWidths.visualization = newWidth;
        newWidths.controls = Math.max(15, remaining);
      }

      return newWidths;
    });
  }, []);

  // Vertical resizing for panels
  const handleVerticalMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isDraggingRef.current = 'vertical';
    startYRef.current = e.clientY;
    startHeightsRef.current = { ...panelHeights };
    document.body.style.cursor = 'row-resize';
    document.body.style.userSelect = 'none';
  }, [panelHeights]);

  const handleVerticalMouseMove = useCallback((e: MouseEvent) => {
    if (isDraggingRef.current !== 'vertical') return;

    const deltaY = e.clientY - startYRef.current;
    const viewportHeight = window.innerHeight - 140; // Account for header and padding
    const deltaPercent = (deltaY / viewportHeight) * 100;

    setPanelHeights(prev => ({
      ...prev,
      visualization: Math.max(30, Math.min(100, startHeightsRef.current.visualization + deltaPercent))
    }));
  }, []);

  const handleMouseUp = useCallback(() => {
    isDraggingRef.current = null;
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
  }, []);

  // Add global mouse event listeners
  useEffect(() => {
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mousemove', handleVerticalMouseMove);
    document.addEventListener('mouseup', handleMouseUp);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mousemove', handleVerticalMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [handleMouseMove, handleMouseUp, handleVerticalMouseMove]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#0A0A0F] to-[#14141F] text-white">
      <Header />

      <div className="flex gap-2 p-6 items-stretch h-[calc(100vh-80px)]">
        {/* Training Panel */}
        <div
          className="flex flex-col h-full overflow-hidden bg-[#0A0A0F]"
          style={{ width: `${panelWidths.training}%` }}
        >
          <div className="flex-1 overflow-hidden">
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
        </div>

        {/* Horizontal Resize Handle - Training/Visualization */}
        <div
          ref={el => dragRefs.current.training = el}
          className="w-2 bg-white/10 hover:bg-white/20 cursor-col-resize flex items-center justify-center group transition-colors"
          onMouseDown={(e) => handleMouseDown(e, 'training')}
        >
          <GripVertical className="w-4 h-4 text-white/40 group-hover:text-white/60 transition-colors" />
        </div>

        {/* Visualization Panel */}
        <div
          className="flex flex-col h-full"
          style={{ width: `${panelWidths.visualization}%` }}
        >
          <div
            className="flex-1 relative"
            style={{ height: `${panelHeights.visualization}%` }}
          >
            <RandomForestVisualization
              trees={trees}
              maxDepth={maxDepth}
              minSamplesSplit={minSamplesSplit}
              minSamplesLeaf={minSamplesLeaf}
              modelState={modelState}
              onTreesChange={setTrees}
              onMaxDepthChange={setMaxDepth}
              onMinSamplesSplitChange={setMinSamplesSplit}
              onMinSamplesLeafChange={setMinSamplesLeaf}
            />

            {/* Vertical Resize Handle */}
            <div
              className="absolute bottom-0 left-0 right-0 h-2 bg-white/10 hover:bg-white/20 cursor-row-resize flex items-center justify-center group transition-colors"
              onMouseDown={handleVerticalMouseDown}
            >
              <GripVertical className="w-4 h-4 text-white/40 group-hover:text-white/60 transition-colors rotate-90" />
            </div>
          </div>
        </div>

        {/* Horizontal Resize Handle - Visualization/Controls */}
        <div
          ref={el => dragRefs.current.visualization = el}
          className="w-2 bg-white/10 hover:bg-white/20 cursor-col-resize flex items-center justify-center group transition-colors"
          onMouseDown={(e) => handleMouseDown(e, 'visualization')}
        >
          <GripVertical className="w-4 h-4 text-white/40 group-hover:text-white/60 transition-colors" />
        </div>

        {/* Controls Panel */}
        <div
          className="flex flex-col h-full overflow-hidden bg-[#0A0A0F]"
          style={{ width: `${panelWidths.controls}%` }}
        >
          <div className="flex-1 overflow-hidden">
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

        {/* Reset Layout Button */}
        <div className="fixed bottom-4 right-4 z-50">
          <button
            onClick={resetPanels}
            className="w-10 h-10 rounded-full bg-white/10 hover:bg-white/20 border border-white/20 flex items-center justify-center transition-colors"
            title="Reset panel layout"
          >
            <Minimize className="w-5 h-5 text-white/60" />
          </button>
        </div>
      </div>
    </div>
  );
}
