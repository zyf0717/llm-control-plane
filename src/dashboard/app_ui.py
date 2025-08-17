import shinyswatch
from shiny import ui

app_ui = ui.page_fluid(
    ui.tags.script(
        """
        document.addEventListener('DOMContentLoaded', () => {
            const ta = document.getElementById('userTextInput');
            const btn = document.getElementById('send');
            if (!ta || !btn) return;

            ta.addEventListener('keydown', (e) => {
                if (e.isComposing) return; // IME
                if (e.key === 'Enter' && e.shiftKey) {
                    e.preventDefault();
                    btn.click();
                }
            });
        });
        """
    ),
    ui.tags.style(
        """
        /* Make the sidebar independently scrollable */
        .sidebar {
            max-height: 100vh;
            overflow-y: auto;
            position: sticky;
            top: 0;
            background: inherit;
        }
        """
    ),
    ui.navset_bar(
        ui.nav_panel(
            "Single-Node",
            ui.page_sidebar(
                ui.sidebar(
                    ui.tags.script(
                        """
                Shiny.addCustomMessageHandler("logout", function(_) {
                    window.location.href = "https://llm-dashboard.paperclips.dev/cdn-cgi/access/logout";
                });
                """
                    ),
                    ui.layout_columns(
                        ui.input_select("endpoint", "", choices=[]),
                        ui.input_action_button("refreshEndpoints", "⟳"),
                        col_widths=[9, 3],
                    ),
                    ui.input_switch("stream", "Streaming", True),
                    ui.input_switch("autoScroll", "Auto-scroll", True),
                    ui.input_switch("outputJSON", "JSON", False),
                    shinyswatch.theme_picker_ui(),
                    ui.hr(),
                    ui.input_action_button("logout", "Logout"),
                ),
                ui.layout_columns(
                    ui.card(
                        ui.layout_columns(
                            ui.input_text(
                                "convoID",
                                "",
                                placeholder="Conversation ID",
                                width="100%",
                            ),
                            ui.input_action_button("generateConvoID", "New"),
                            col_widths=[7, 5],
                        ),
                        ui.input_text_area(
                            "userTextInput",
                            "",
                            rows=6,
                            placeholder="Ask anything",
                            width="100%",
                        ),
                        ui.input_task_button(
                            "send", "Send (Shift + Enter)", auto_reset=False
                        ),
                        ui.output_ui("outputRunInfo"),
                    ),
                    ui.output_ui("responseBox"),
                    col_widths=[3, 9],
                    fillable=False,
                ),
            ),
        ),
        ui.nav_panel("Multi-Node"),
        ui.nav_panel(
            "History",
            ui.layout_columns(
                ui.input_action_button("refreshHistory", "Refresh History"),
                col_widths=[3],
            ),
            ui.output_ui("historyBox"),
        ),
        title="LLM Control Plane",
    ),
    theme=shinyswatch.theme.flatly,
)
