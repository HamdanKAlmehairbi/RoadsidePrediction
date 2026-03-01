

# SEAL Dashboard — Implementation Plan

## Overview
A dark-mode React dashboard for a federated reinforcement learning traffic light control system. Desktop-first, using Tailwind CSS and Recharts with all mock/hardcoded data.

## Design System
- **Background:** #0f1117, **Cards:** #1a1d27, **Borders:** #2d2f3e
- **Accent colors:** Teal (#14b8a6) for FedRL, Amber (#f59e0b) for MARL, Purple (#a855f7) for SARL, Blue (#3b82f6) for primary actions
- Monospace font for all live metric values
- Collapsible sidebar + top nav bar on all pages

## Layout & Navigation
- **Top nav bar:** "SEAL" logo left, links right: Dashboard / Simulation / Compare / Training / Communication
- **Collapsible left sidebar:** Icon-only when collapsed (w-14), 240px expanded, mirroring nav links with icons
- Toast notifications bottom-right for key events

## Pages

### 1. Dashboard (`/`)
- **Hero:** Large heading "Smart Edge-Enabled Adaptive Traffic Lights" with subheading about 36% communication reduction
- **4 stat cards:** Communication Reduction (36.24%, teal), Reward vs MARL (−2.11%, amber), Travel Time Improvement (18.14%, blue), Active Intersections (49, purple) — each with an appropriate icon
- **"How It Works" section:** 3 cards — Simulate, Learn, Federate — explaining the pipeline
- **2 CTA buttons:** "Watch a Simulation" (teal → /simulation), "Start Training" (blue → /training)

### 2. Simulation (`/simulation`)
- **Controls bar:** Policy dropdown, topology dropdown, seed input, Run button, play/pause, speed buttons (1×/2×/5×)
- **Main area:** Static placeholder canvas showing a bird's-eye 3×3 grid with road rectangles, colored traffic light circles, and vehicle dots (placeholder for future WebSocket-driven animation)
- **Right sidebar (280px):** Live metric cards (Halted Vehicles, Mean Speed, Mean Reward with sparkline, Timestep) + cumulative reward line chart (360 mock data points)

### 3. Compare (`/compare`)
- **Controls bar:** Policy A dropdown, Policy B dropdown, topology dropdown, seed input, "Run Both" button
- **Split-screen:** Two simulation canvas placeholders side-by-side with policy label chips in accent colors
- **Center diff panel:** Side-by-side metric comparison (Halted, Speed, Reward) highlighting the better value in green, with a verdict badge
- **Bottom chart:** Dual-line Recharts chart comparing cumulative reward over 360 timesteps

### 4. Training (`/training`)
- **Controls:** Trainer dropdown, topology dropdown, ranked toggle, episodes slider (10–100), Start Training button, "Show existing results" toggle
- **Main chart:** Multi-line Recharts chart (episodes vs mean reward) with 3 mock trainer lines, dashed vertical federation round markers, hover tooltips, and a pulsing dot for active training
- **Status bar:** Episode progress, current reward, federation round status with progress bar (visible during training)
- **Summary table:** Sortable table with Trainer, Final Reward, Best Reward, Episodes, Aggregation Function
- **Post-training CTA:** "Simulate this policy →" button

### 5. Communication (`/communication`)
- **Banner:** Full-width teal gradient with key finding about 36.24% communication reduction
- **Left column:** Stacked bar chart — messages per episode by type (Policy, Observation, Rank, Action, Vehicle) across all trainers
- **Right column:** Scatter chart — reward vs communication cost with labeled points and dashed Pareto frontier
- **Explainer card:** 5 bullets explaining each message type and its bandwidth cost
- **Raw data table:** Full breakdown of message counts and rewards per trainer

## Technical Notes
- All data is mock/hardcoded with TODO comments for future API integration
- React Router for navigation across 5 pages
- Recharts for all charts (line, bar, scatter, sparkline)
- Custom dark theme applied via Tailwind CSS variables
- Collapsible sidebar using shadcn Sidebar component

