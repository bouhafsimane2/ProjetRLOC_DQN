@echo off
cd /d "%~dp0.."
tensorboard --logdir_spec v2_local:runs,v1_colab:colab_results/runs
pause
