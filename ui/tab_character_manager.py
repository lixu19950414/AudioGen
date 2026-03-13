"""Tab 4 — 角色管理"""

from __future__ import annotations

import gradio as gr

from ui.common import (
    character_display_rows,
    design_character_display_rows,
    load_characters,
    load_design_characters,
    save_characters,
    save_design_characters,
)
from core.app_logger import log_event, EVENT_CHAR_REMOVE


def tab_character_manager():
    with gr.Tab("角色管理"):
        with gr.Tabs():
            # --- 子 Tab 1：克隆角色管理 ---
            with gr.Tab("克隆角色"):
                char_table = gr.Dataframe(
                    headers=["角色名", "参考音频路径", "参考文字", "描述"],
                    datatype=["str", "str", "str", "str"],
                    label="克隆角色列表",
                    interactive=False,
                    value=character_display_rows(load_characters()),
                )
                with gr.Row():
                    refresh_btn = gr.Button("刷新")
                    delete_clone_name = gr.Textbox(label="角色名", placeholder="输入要删除的角色名", scale=2)
                    delete_clone_btn = gr.Button("删除", variant="stop")
                clone_op_status = gr.Textbox(label="操作状态", interactive=False)

            # --- 子 Tab 2：设计角色管理 ---
            with gr.Tab("设计角色"):
                design_table = gr.Dataframe(
                    headers=["角色名", "音色描述", "描述"],
                    datatype=["str", "str", "str"],
                    label="设计角色列表",
                    interactive=False,
                    value=design_character_display_rows(load_design_characters()),
                )
                with gr.Row():
                    design_refresh_btn = gr.Button("刷新")
                    delete_design_name = gr.Textbox(label="角色名", placeholder="输入要删除的角色名", scale=2)
                    delete_design_btn = gr.Button("删除", variant="stop")
                design_op_status = gr.Textbox(label="操作状态", interactive=False)

        def delete_clone_char(name):
            name = name.strip()
            if not name:
                return character_display_rows(load_characters()), "请输入角色名"
            chars = load_characters()
            if name in chars:
                del chars[name]
                save_characters(chars)
                log_event(EVENT_CHAR_REMOVE, f"克隆角色={name}")
                return character_display_rows(chars), f"已删除克隆角色：{name}"
            return character_display_rows(chars), f"角色不存在：{name}"

        def delete_design_char(name):
            name = name.strip()
            if not name:
                return design_character_display_rows(load_design_characters()), "请输入角色名"
            chars = load_design_characters()
            if name in chars:
                del chars[name]
                save_design_characters(chars)
                log_event(EVENT_CHAR_REMOVE, f"设计角色={name}")
                return design_character_display_rows(chars), f"已删除设计角色：{name}"
            return design_character_display_rows(chars), f"角色不存在：{name}"

        refresh_btn.click(
            fn=lambda: character_display_rows(load_characters()),
            outputs=[char_table],
        )
        delete_clone_btn.click(
            fn=delete_clone_char,
            inputs=[delete_clone_name],
            outputs=[char_table, clone_op_status],
        )
        design_refresh_btn.click(
            fn=lambda: design_character_display_rows(load_design_characters()),
            outputs=[design_table],
        )
        delete_design_btn.click(
            fn=delete_design_char,
            inputs=[delete_design_name],
            outputs=[design_table, design_op_status],
        )

    return char_table, design_table, delete_clone_btn, delete_design_btn
