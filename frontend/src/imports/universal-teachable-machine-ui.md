Create a complete high-fidelity UI design for "MindCraft" - a dark-themed futuristic AI dashboard where users upload data and train Random Forest models. Style: dark background (#0A0A0F to #14141F), glass-morphism panels, neon green (#39FF14) accents for Random Forest, electric blue (#00F0FF) accents for MobileNet features.

LAYOUT STRUCTURE:
Left Panel (30%): Training Input
Middle Panel (45%): Model Visualization
Right Panel (25%): Controls & Metrics
Bottom (full width): Test Prediction

LEFT PANEL - TRAINING INPUT:
Two toggle tabs: [Tabular] [Image]

TABULAR VIEW:
- "Choose File" button
- File name: "Framingham Dataset.csv"
- "Features" header with searchable checkbox list:
  ☐ AGE  ☐ SEX  ☐ TOTCHOL  ☐ SYSBP  ☐ GLUCOSE  ☐ BMI  ☐ CIGPDAY  ☐ HEARTRATE
- "Target Column" dropdown [TenYearCHD ▼]
- Mode selector: [Classification] [Regression]
- "TRAIN RANDOM FOREST" button (neon green)

IMAGE VIEW:
- "Upload ZIP files (one per class)"
- Two file cards:
  - "cats.zip" with image icon + green check
  - "dogs.zip" with image icon + green check
- "Add Class" button with plus
- Note: "Classes extracted from zip filenames"
- "TRAIN IMAGE CLASSIFIER" button (neon green)
- Small info: "Using MobileNetV2 feature extractor"

MIDDLE PANEL - MODEL VISUALIZATION:
Top: "RANDOM FOREST ACTIVE" badge with green glow

VISUALIZATION:
Abstract glowing node grid pattern exactly like this:
    O     O     O     O
       O     O     O
    O     O     O     O
       O     O     O
    O     O     O     O

Nodes are glowing green circles (#39FF14) arranged in offset grid. Some nodes brighter than others. Subtle connecting lines between some nodes with flowing light effect. Dark background with faint particle effects. Looks like constellation/star pattern.

Bottom of panel: Small feature importance bars:
[████░░░░░] AGE: 0.32
[██░░░░░░░] TOTCHOL: 0.18
[█░░░░░░░░] GLUCOSE: 0.12
[░░░░░░░░░] SEX: 0.08

RIGHT PANEL - CONTROLS & METRICS:
Three glass cards:

TOP CARD "Model Info":
- "Random Forest Classifier"
- Number of trees: 100
- Max depth: 10
- Features: 8
- Classes: 2 (binary)

MIDDLE CARD "Performance":
- Training accuracy: 94%
- Progress ring showing 94% full
- "Training Complete" badge with checkmark
- OOB score: 0.89

BOTTOM CARD "Parameters":
- Trees slider: [100] (10-500)
- Max depth slider: [10] (3-30)
- Min samples split slider: [2]
- Feature sampling: [sqrt ▼]

BOTTOM PANEL - TEST PREDICTION:
Full-width glass panel. Two modes:

TABULAR PREDICTION VIEW:
Dynamic form fields in row:
AGE [45] | SEX [M ▼] | TOTCHOL [240] | SYSBP [135] | GLUCOSE [95]
Large green "PREDICT" button

IMAGE PREDICTION VIEW (alternate):
"Upload Image to Classify" with choose file button
Image preview thumbnail

PREDICTION RESULTS CARD:
"Prediction: At Risk (78% confidence)"
Progress bar at 78%
Small "Feature Importance for this prediction" popup showing which features influenced decision

HEADER:
"MindCraft" with brain icon
Status: "All in-browser • Private"

COLORS:
- Background: #0A0A0F to #14141F gradient
- Random Forest green: #39FF14 (neon green)
- MobileNet/accents: #00F0FF (electric blue)
- Glass panels: rgba(20,20,30,0.7) with backdrop blur
- Text: White, light gray
- All buttons, sliders, interactive elements have glow effects

Generate complete polished UI with all elements exactly as described.