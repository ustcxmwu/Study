from ortools.linear_solver import pywraplp



if __name__ == '__main__':
    # 1. 初始化测试数据 (5个物品, 5个背包)
    # weights[i][j]: 物品 i 放入背包 j 的重量
    weights = [
        [6, 2, 4, 3, 5],  # 物品 0
        [3, 5, 2, 8, 4],  # 物品 1
        [5, 4, 6, 2, 3],  # 物品 2
        [4, 3, 5, 7, 2],  # 物品 3
        [2, 6, 3, 4, 5],  # 物品 4
    ]

    # values[i][j]: 物品 i 放入背包 j 的价值
    values = [
        [15, 12, 10, 14, 11],  # 物品 0
        [10, 18, 12, 15, 13],  # 物品 1
        [14, 13, 16, 11, 15],  # 物品 2
        [12, 11, 15, 20, 14],  # 物品 3
        [11, 14, 11, 13, 16],  # 物品 4
    ]

    capacities = [8, 9, 7, 10, 8]  # 5个背包的各自容量限制

    num_items = len(weights)
    num_bins = len(capacities)

    # 2. 创建 SCIP 混合整数规划求解器
    solver = pywraplp.Solver.CreateSolver("SCIP")

    # 3. 创建决策变量 x[i, j] -> 0 或 1
    x = {}
    for i in range(num_items):
        for j in range(num_bins):
            x[i, j] = solver.IntVar(0, 1, f"x_{i}_{j}")

    # 4. 约束条件 1: 每个背包的重量不能超载
    for j in range(num_bins):
        solver.Add(
            sum(x[i, j] * weights[i][j] for i in range(num_items)) <= capacities[j]
        )

    # 5. 约束条件 2: 每个物品最多只能放进一个背包
    for i in range(num_items):
        solver.Add(sum(x[i, j] for j in range(num_bins)) <= 1)

    # 6. 目标函数: 最大化总价值
    objective = solver.Objective()
    for i in range(num_items):
        for j in range(num_bins):
            objective.SetCoefficient(x[i, j], values[i][j])
    objective.SetMaximization()

    # 7. 求解
    status = solver.Solve()

    # 8. 打印结果
    if status == pywraplp.Solver.OPTIMAL:
        print(f"【OR-Tools】全局最优解总价值: {solver.Objective().Value()}")
        for j in range(num_bins):
            used_weight = sum(
                x[i, j].solution_value() * weights[i][j] for i in range(num_items)
            )
            print(f"  背包 {j} (容量 {capacities[j]}, 已用 {used_weight}):")
            for i in range(num_items):
                if x[i, j].solution_value() > 0.5:
                    print(
                        f"    -> 装入物品 {i} (重量: {weights[i][j]}, 价值: {values[i][j]})"
                    )
    else:
        print("未找到可行解")