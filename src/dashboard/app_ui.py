import shinyswatch
from shiny import ui


def input_action_row(
    input_control,
    button_id,
    button_label,
    *,
    col_widths=None,
    button_class=None,
):
    """Render one input control with a trailing aligned action button."""
    return ui.layout_columns(
        input_control,
        ui.div(
            ui.input_action_button(
                button_id,
                button_label,
                class_=button_class,
            ),
            style="display: flex; align-items: flex-end; height: 100%;",
        ),
        col_widths=col_widths or [10, 2],
    )


app_ui = ui.page_fluid(
    ui.tags.style("""
        :root {
            --dashboard-panel-height: calc(100dvh - 8rem);
            --dashboard-chat-height: var(--dashboard-panel-height);
        }

        /* Make the sidebar independently scrollable */
        .sidebar {
            max-height: 100vh;
            overflow-y: auto;
            position: sticky;
            top: 0;
            background: inherit;
        }

        .dashboard-pane {
            height: var(--dashboard-panel-height);
            min-height: 0;
        }

        .dashboard-side-panel {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .dashboard-run-info {
            flex: 1 1 auto;
            min-height: 0;
            overflow-y: auto;
        }

        .dashboard-square-button {
            width: 2.375rem;
            height: 2.375rem;
            min-width: 2.375rem;
            padding: 0;
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }

        .dashboard-full-width-action {
            width: 100%;
        }

        .dashboard-workflow-upload-file .shiny-label-null {
            display: none;
        }

        .dashboard-workflow-upload-file .form-group {
            margin-bottom: 0;
        }

        .dashboard-trace-json {
            max-height: 28rem;
            overflow: auto;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            margin-bottom: 0;
        }

        @media (max-width: 991.98px) {
            :root {
                --dashboard-panel-height: auto;
                --dashboard-chat-height: min(60dvh, 32rem);
            }

            .dashboard-run-info {
                max-height: 20rem;
            }
        }
        """),
    ui.navset_bar(
        ui.nav_panel(
            "Single-Node",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.tags.script("""
                Shiny.addCustomMessageHandler("logout", function(_) {
                    window.location.href = "https://llm-dashboard.paperclips.dev/cdn-cgi/access/logout";
                });
                """),
                    input_action_row(
                        ui.input_select(
                            "endpoint",
                            "Machine Endpoint",
                            choices=[],
                            width="100%",
                        ),
                        "refreshEndpoints",
                        "🔄",
                        button_class="dashboard-square-button",
                    ),
                    ui.input_switch("stream", "Streaming", True),
                    ui.input_switch("autoScroll", "Auto-scroll", True),
                    ui.input_switch("outputJSON", "JSON output", False),
                    ui.input_switch("outputReasoning", "Show reasoning", True),
                    ui.input_select(
                        "reasoningEffort",
                        "Reasoning Level",
                        choices={
                            "none": "None",
                            "low": "Low",
                            "medium": "Medium",
                            "high": "High",
                        },
                        selected="medium",
                    ),
                    ui.output_ui("system_prompt_ui"),
                    ui.hr(),
                    shinyswatch.theme_picker_ui(),
                    ui.input_action_button("logout", "Logout"),
                    width="320px",
                ),
                ui.layout_columns(
                    ui.chat_ui(
                        "chat",
                        width="100%",
                        height="var(--dashboard-chat-height)",
                    ),
                    ui.div(
                        input_action_row(
                            ui.input_text(
                                "convoID",
                                "Conversation ID",
                                placeholder="Convo ID",
                                width="100%",
                            ),
                            "generateConvoID",
                            "🔄",
                            button_class="dashboard-square-button",
                        ),
                        input_action_row(
                            ui.input_select(
                                "ragEndpoint",
                                "RAG Endpoint",
                                choices=[],
                                width="100%",
                            ),
                            "refreshRagEndpoints",
                            "🔄",
                            button_class="dashboard-square-button",
                        ),
                        ui.input_select(
                            "searchProvider",
                            "Search Provider",
                            choices=[],
                            width="100%",
                        ),
                        ui.output_ui("file_upload_ui"),
                        ui.div(
                            ui.output_ui("outputRunInfo"),
                            class_="dashboard-run-info",
                        ),
                        class_="dashboard-pane dashboard-side-panel",
                    ),
                    col_widths=[9, 3],
                ),
                fillable=True,
            ),
        ),
        ui.nav_panel(
            "Multi-Node",
            ui.div("Multi-node UI coming soon!"),
        ),
        ui.nav_panel(
            "Workflows",
            ui.layout_columns(
                ui.div(
                    input_action_row(
                        ui.input_select(
                            "workflowSelector",
                            "Workflow",
                            choices={},
                            width="100%",
                        ),
                        "refreshWorkflows",
                        "Refresh",
                        col_widths=[8, 4],
                    ),
                    ui.output_ui("workflowSpecDetails"),
                    ui.input_text_area(
                        "workflowParams",
                        "Params JSON",
                        value='{\n  "goal": ""\n}',
                        rows=8,
                        width="100%",
                    ),
                    ui.layout_columns(
                        ui.input_select(
                            "workflowEndpoint",
                            "Endpoint",
                            choices={},
                            width="100%",
                        ),
                        ui.input_select(
                            "workflowReasoning",
                            "Reasoning",
                            choices={
                                "": "Default",
                                "low": "Low",
                                "medium": "Medium",
                                "high": "High",
                            },
                            selected="",
                            width="100%",
                        ),
                        col_widths=[7, 5],
                    ),
                    ui.input_text(
                        "workflowConvoID",
                        "Conversation ID",
                        placeholder="Optional",
                        width="100%",
                    ),
                    ui.input_select(
                        "workflowRagEndpoint",
                        "RAG Endpoint",
                        choices={},
                        width="100%",
                    ),
                    ui.input_select(
                        "workflowSearchProvider",
                        "Search Provider",
                        choices={},
                        width="100%",
                    ),
                    ui.output_ui("workflow_file_upload_ui"),
                    ui.input_action_button(
                        "createWorkflowRun",
                        "Create run",
                        class_="btn-primary dashboard-full-width-action",
                    ),
                ),
                ui.div(
                    input_action_row(
                        ui.input_select(
                            "workflowRunSelector",
                            "Run",
                            choices={},
                            width="100%",
                        ),
                        "refreshWorkflowRuns",
                        "Refresh",
                        col_widths=[8, 4],
                    ),
                    ui.layout_columns(
                        ui.input_action_button("advanceWorkflowRun", "Run next step"),
                        ui.input_action_button(
                            "runWorkflowToCompletion", "Run to completion"
                        ),
                        input_action_row(
                            ui.input_text(
                                "workflowRetryStepID",
                                "",
                                placeholder="failed step id",
                                width="100%",
                            ),
                            "retryWorkflowStep",
                            "Retry step",
                            col_widths=[7, 5],
                        ),
                        col_widths=[3, 3, 6],
                    ),
                    ui.output_ui("workflowRunDetails"),
                ),
                col_widths=[4, 8],
            ),
        ),
        ui.nav_panel(
            "History",
            input_action_row(
                ui.input_select(
                    "historyConvoSelector",
                    "Conversation",
                    choices={},
                    width="100%",
                ),
                "refreshHistory",
                "Refresh",
                col_widths=[3, 2],
            ),
            ui.output_ui("historyBox"),
        ),
        ui.nav_panel(
            "Traces",
            ui.layout_columns(
                ui.input_text(
                    "traceConvoFilter",
                    "Convo ID",
                    placeholder="Optional",
                    width="100%",
                ),
                ui.input_text(
                    "traceIDFilter",
                    "Trace ID",
                    placeholder="Optional",
                    width="100%",
                ),
                ui.input_text(
                    "traceEndpointFilter",
                    "Endpoint",
                    placeholder="Optional",
                    width="100%",
                ),
                ui.input_select(
                    "traceMaxRows",
                    "Rows",
                    choices={"50": "50", "100": "100", "200": "200", "500": "500"},
                    selected="200",
                    width="100%",
                ),
                ui.div(
                    ui.input_action_button("refreshTraces", "Refresh"),
                    style="display: flex; align-items: flex-end; height: 100%;",
                ),
                col_widths=[3, 3, 3, 1, 2],
            ),
            ui.output_ui("traceBox"),
        ),
        title="LLM Control Plane",
    ),
    theme=shinyswatch.theme.flatly,
)
