@echo off
title Mini VLA 一键演示
cd /d "%~dp0"

echo ============================================================
echo   Mini VLA 一键演示 (总计约 2-3 分钟, CPU 即可)
echo ============================================================
echo.
echo [1/3] 运行评估: 语言响应 gap / CFG 扫描 ...
python evaluate.py --ckpt ckpt\vla_best.pt
if errorlevel 1 goto :err

echo.
echo [2/3] 生成静态图: 轨迹图 / 动作柱状图 / 世界模型闭环 ...
python visualize.py --ckpt ckpt\vla_best.pt --out figs
if errorlevel 1 goto :err

echo.
echo [3/3] 生成演示视频: 轨迹动画 / 第一视角对比 ...
python make_video.py
if errorlevel 1 goto :err

echo.
echo ============================================================
echo   全部完成! 成果位置:
echo     指标 JSON   - eval_results\metrics.json
echo     静态图      - figs\
echo     演示视频    - videos\
echo ============================================================
pause
exit /b 0

:err
echo.
echo [错误] 上一步运行失败, 请检查上方报错信息
pause
exit /b 1
