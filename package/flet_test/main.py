import flet as ft


# 1. 通用卡牌工厂（支持横向/纵向、正面/背面）
def create_game_card(name="", cost=0, attack=0, defense=0, desc="", is_face_up=True, is_vertical=False, on_click=None,
                     scale=1.0):
    if not is_face_up:
        if is_vertical:
            # 侧边玩家的竖向麻将/卡牌背面
            card_content = ft.Container(
                width=35 * scale, height=75 * scale,
                bgcolor=ft.Colors.INDIGO_900,
                border=ft.Border.all(1.5, ft.Colors.INDIGO_300),
                border_radius=4,
                alignment=ft.alignment.Alignment(0, 0),
                content=ft.Text("🎴", size=10)
            )
        else:
            # 对家/队友的横向卡牌背面
            card_content = ft.Container(
                width=70 * scale, height=100 * scale,
                bgcolor=ft.Colors.INDIGO_900,
                border=ft.Border.all(2, ft.Colors.INDIGO_300),
                border_radius=6,
                alignment=ft.alignment.Alignment(0, 0),
                content=ft.Text("🎴 牌背", size=11, color=ft.Colors.with_opacity(0.6, ft.Colors.WHITE))
            )
    else:
        # 正面明牌（主视角）
        card_content = ft.Container(
            width=85 * scale, height=120 * scale,
            bgcolor=ft.Colors.AMBER_50,
            border=ft.Border.all(2, ft.Colors.AMBER_900),
            border_radius=6,
            padding=4,
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Container(
                                content=ft.Text(str(cost), color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD, size=9),
                                bgcolor=ft.Colors.BLUE_700, width=16, height=16, border_radius=8,
                                alignment=ft.alignment.Alignment(0, 0),
                            ),
                            ft.Text(name, weight=ft.FontWeight.BOLD, size=10, color=ft.Colors.BROWN_900),
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                    ft.Container(
                        content=ft.Icon(ft.Icons.SHIELD_ROUNDED, size=18, color=ft.Colors.AMBER_800),
                        bgcolor=ft.Colors.AMBER_200, expand=True, border_radius=4,
                        alignment=ft.alignment.Alignment(0, 0),
                    ),
                    ft.Container(
                        content=ft.Text(desc, size=8, color=ft.Colors.GREY_900, italic=True),
                        height=20,
                    ),
                    ft.Row(
                        [
                            ft.Text(f"⚔️{attack}", size=9, weight=ft.FontWeight.BOLD, color=ft.Colors.RED_700),
                            ft.Text(f"🛡️{defense}", size=9, weight=ft.FontWeight.BOLD, color=ft.Colors.GREEN_700),
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                ],
                spacing=1,
            ),
        )

    if on_click and is_face_up:
        return ft.GestureDetector(content=card_content, on_tap=on_click)
    return card_content


def main(page: ft.Page):
    page.title = "Flet 4人麻将式环形桌游原型"
    page.window_width = 1050
    page.window_height = 850
    page.bgcolor = ft.Colors.BLUE_GREY_900
    page.padding = 15

    status_text = ft.Text("回合进行中：轮到【南家 (你)】出牌", color=ft.Colors.with_opacity(0.8, ft.Colors.WHITE), size=12)

    # ----------------- 1. 北家 (上方对家) -----------------
    north_hand = ft.Row([create_game_card(is_face_up=False, scale=0.8) for _ in range(3)], spacing=6,
                        alignment=ft.MainAxisAlignment.CENTER)
    north_panel = ft.Container(
        content=ft.Column([
            ft.Text("👤 北家 (对家) - 手牌: 3", color=ft.Colors.RED_300, size=11, weight=ft.FontWeight.BOLD),
            north_hand
        ], alignment=ft.MainAxisAlignment.CENTER, spacing=2),
        alignment=ft.alignment.Alignment(0, 0)
    )

    # ----------------- 2. 西家 (左侧) 与 东家 (右侧) -----------------
    west_hand = ft.Column([create_game_card(is_face_up=False, is_vertical=True, scale=0.85) for _ in range(3)],
                          spacing=4, alignment=ft.MainAxisAlignment.CENTER)
    west_panel = ft.Container(
        content=ft.Column([
            ft.Text("👤 西家", color=ft.Colors.ORANGE_300, size=10, weight=ft.FontWeight.BOLD),
            west_hand
        ], alignment=ft.MainAxisAlignment.CENTER, spacing=4),
        padding=8,
        bgcolor=ft.Colors.with_opacity(0.15, ft.Colors.BLACK),
        border_radius=8,
        width=75,
    )

    east_hand = ft.Column([create_game_card(is_face_up=False, is_vertical=True, scale=0.85) for _ in range(3)],
                          spacing=4, alignment=ft.MainAxisAlignment.CENTER)
    east_panel = ft.Container(
        content=ft.Column([
            ft.Text("👤 东家", color=ft.Colors.ORANGE_300, size=10, weight=ft.FontWeight.BOLD),
            east_hand
        ], alignment=ft.MainAxisAlignment.CENTER, spacing=4),
        padding=8,
        bgcolor=ft.Colors.with_opacity(0.15, ft.Colors.BLACK),
        border_radius=8,
        width=75,
    )

    # ----------------- 3. 中央共享战场区 (牌河 / 桌面) -----------------
    battlefield_stack = ft.Stack(
        controls=[
            ft.Container(
                width=720, height=280,
                bgcolor=ft.Colors.GREEN_800,
                border_radius=16,
                border=ft.Border.all(3, ft.Colors.GREEN_900),
                alignment=ft.alignment.Alignment(0, 0),
                content=ft.Text("【 中 央 牌 河 / 战 场 区 】", color=ft.Colors.with_opacity(0.25, ft.Colors.WHITE),
                                size=15, weight=ft.FontWeight.BOLD)
            )
        ],
        width=720, height=280,
    )

    # ----------------- 4. 南家 (下方主视角我方) -----------------
    my_hand_row = ft.Row(spacing=10, alignment=ft.MainAxisAlignment.CENTER)

    def play_my_card(card_info):
        status_text.value = f"你打出了：【{card_info['name']}】"
        status_text.update()

        # 在中央战场随机或依次生成明牌打出的卡牌
        card_on_board = create_game_card(**card_info, is_face_up=True, scale=0.8)
        card_on_board.left = 180 + (len(battlefield_stack.controls) - 1) * 60
        card_on_board.top = 70

        battlefield_stack.controls.append(card_on_board)
        battlefield_stack.update()

    initial_hand = [
        {"name": "关羽", "cost": 4, "attack": 4, "defense": 3, "desc": "【武圣】红色牌当杀。"},
        {"name": "赵云", "cost": 3, "attack": 3, "defense": 3, "desc": "【龙胆】杀闪互换。"},
        {"name": "诸葛亮", "cost": 3, "attack": 1, "defense": 3, "desc": "【观星】看牌堆顶。"},
    ]

    for info in initial_hand:
        def make_cb(data=info):
            return lambda _: play_my_card(data)

        my_hand_row.controls.append(create_game_card(**info, is_face_up=True, on_click=make_cb(), scale=0.85))

    south_panel = ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Text("⭐ 南家 (你 - 主视角)", color=ft.Colors.GREEN_300, size=12, weight=ft.FontWeight.BOLD),
                status_text,
                ft.Text("❤️ 4/5", color=ft.Colors.with_opacity(0.7, ft.Colors.WHITE), size=11)
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            my_hand_row
        ], spacing=4),
        padding=10,
        bgcolor=ft.Colors.with_opacity(0.2, ft.Colors.BLACK),
        border_radius=10,
        width=880,
    )

    # ----------------- 整体三层环形布局组装 -----------------
    page.add(
        ft.Column(
            [
                # 第一层：北家
                north_panel,
                ft.Container(height=5),
                # 第二层：西家 + 中央战场 + 东家
                ft.Row(
                    [
                        west_panel,
                        battlefield_stack,
                        east_panel,
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    spacing=10,
                ),
                ft.Container(height=5),
                # 第三层：南家 (我方)
                south_panel,
            ],
            spacing=0,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )
    )


if __name__ == "__main__":
    ft.run(main)