# 错题相关功能代码整理

本目录用于单独上传和合并以下四个模块相关代码：

- 错题概览
- 生成题目
- 生成试卷
- 试卷和解析

## 文件结构

```text
cuoti/
  app/
    database.py
    generation.py
    main.py
    models.py
    question_service.py
  static/
    index.html
    app.js
    styles.css
  requirements.txt
```

## 合并重点

- `static/index.html`：左侧导航、错题概览、生成题目、生成试卷、试卷和解析页面结构。
- `static/app.js`：筛选错题、生成类似题、生成完整试卷、保存生成试卷、试卷解析详情、PDF 打印导出逻辑。
- `static/styles.css`：以上页面和打印导出相关样式。
- `app/main.py`：生成题目接口、保存/读取生成试卷接口、题目图片接口等。
- `app/generation.py`：调用 GPT 生成类似题，支持文字和真实 SVG 图形分离。
- `app/models.py`：包含生成试卷保存表 `GeneratedPaper`。
- `app/question_service.py`：题目录入、错题判断、图片保存等基础逻辑。
- `app/database.py`：数据库连接和建表初始化逻辑。

## 注意

如果目标仓库已有同名文件，不建议直接整体覆盖；建议按模块对比合并，尤其是 `static/app.js` 和 `app/main.py`。
