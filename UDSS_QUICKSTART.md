# UDSS 快速启动

## 方式一：一键启动（推荐）

```bash
# Windows
start_udss.bat

# Linux/Mac
bash start_udss.sh
```

## 方式二：手动启动

**终端1 - 启动后端API**
```bash
python udss_api.py
```

**终端2 - 启动前端**
```bash
cd frontend
npm run dev
```

## 访问

- 前端页面: http://localhost:5173
- API健康检查: http://localhost:5000/api/health

## 使用流程

1. 打开 http://localhost:5173
2. 在输入框中输入您的问题（如"我应该选择哪个offer"）
3. 点击"提交场景"
4. 系统将：
   - 搜索最新相关信息
   - 通过数学公式计算最优决策
   - 生成可执行方案
5. 可以提供反馈帮助系统学习

## 核心公式

```
Decision(E, Q, R) = argmax_a [α·V(a) + β·Sim(R,a) - γ·C(a)]

其中：
- E = 环境向量
- Q = 问题向量
- R = 搜索结果
- V(a) = 动作价值
- Sim(R,a) = 相似度
- C(a) = 执行成本
- α, β, γ = 可学习权重
```

## API端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/health` | GET | 健康检查 |
| `/api/scenario` | POST | 执行决策 |
| `/api/feedback` | POST | 提供反馈 |
| `/api/custom-scenario` | POST | 决策+验证 |
| `/api/learning-summary` | GET | 学习统计 |
| `/api/reset-learning` | POST | 重置学习 |