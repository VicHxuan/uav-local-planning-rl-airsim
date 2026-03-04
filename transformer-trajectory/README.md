# Migration Notice: AirSim → Isaac Sim / Isaac Lab

## Background

Starting from **2026.1.1**, this project **stops using AirSim**.

The primary reason is that **AirSim is difficult to scale for parallel execution** in our current workflow, which makes large-scale training and evaluation inefficient.

## New Direction

From **2026.1.1** onward, we begin migrating the simulation stack to:

- **Isaac Sim 4.5**
- **Isaac Lab 2.1.0**

The goal of this migration is to achieve **better parallelization**, improve simulation throughput, and provide a more maintainable long-term simulation foundation.

## Status

- AirSim integration: **deprecated / no longer used**
- Isaac Sim + Isaac Lab integration: **in progress**
- Existing features will be gradually re-implemented and validated on the new stack.

## Notes

- During the transition period, some components may be temporarily unavailable or behave differently compared with the AirSim version.
- Please refer to the latest commits and documentation updates for migration progress and usage changes.
