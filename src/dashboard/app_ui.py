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

        .dashboard-step-row {
            display: grid;
            gap: 0.5rem;
            align-items: stretch;
            margin-bottom: 0.75rem;
        }

        .dashboard-step-row.cols-1 { grid-template-columns: repeat(1, minmax(0, 1fr)); }
        .dashboard-step-row.cols-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .dashboard-step-row.cols-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
        .dashboard-step-row.cols-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }

        .dashboard-step-button {
            min-width: 0;
            border: 0;
            padding: 0;
            background: var(--bs-body-bg);
            color: var(--bs-body-color);
            text-align: left;
            appearance: none;
        }

        .dashboard-step-button:focus {
            outline: none;
        }

        .dashboard-step-button:focus-visible {
            outline: 2px solid var(--bs-primary);
            outline-offset: 2px;
        }

        .dashboard-step-button.is-active .dashboard-step-box {
            box-shadow: 0 0 0 2px var(--dashboard-step-color, var(--bs-primary));
        }

        .dashboard-step-panel {
            width: 100%;
            border: 0;
            background: transparent;
            color: var(--bs-body-color);
            padding: 0;
            margin-bottom: 0.75rem;
        }

        .dashboard-step-detail-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.75rem;
        }

        .dashboard-step-detail-box {
            border: 1px solid var(--bs-border-color);
            border-radius: var(--bs-border-radius);
            background: var(--bs-body-bg);
            color: var(--bs-body-color);
            overflow: hidden;
            min-width: 0;
        }

        .dashboard-step-detail-title {
            font-weight: 700;
            padding: 0.5rem 0.75rem;
            border-bottom: 1px solid var(--bs-border-color);
            background: var(--bs-body-bg);
            color: var(--bs-body-color);
        }

        .dashboard-step-detail-body {
            max-height: 28rem;
            overflow: auto;
            padding: 0.75rem;
            background: var(--bs-body-bg);
            color: var(--bs-body-color);
        }

        .dashboard-step-detail-body .dashboard-trace-json {
            background: transparent !important;
            color: inherit !important;
            padding: 0;
            max-height: none;
            overflow: visible;
        }

        .dashboard-step-empty {
            margin: 0;
            color: var(--bs-secondary-color);
        }

        .dashboard-trace-json {
            max-height: 28rem;
            overflow: auto;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            margin-bottom: 0;
            background-color: transparent !important;
            color: var(--bs-emphasis-color) !important;
            border: 0 !important;
        }

        .dashboard-workflow-disabled {
            position: relative;
            opacity: 0.58;
        }

        .dashboard-workflow-disabled label {
            color: var(--bs-secondary-color) !important;
        }

        .dashboard-workflow-disabled .selectize-control,
        .dashboard-workflow-disabled .selectize-input,
        .dashboard-workflow-disabled select {
            cursor: not-allowed !important;
            filter: grayscale(1);
        }

        .dashboard-workflow-disabled .selectize-input,
        .dashboard-workflow-disabled .form-select {
            background-color: var(--bs-secondary-bg) !important;
            border-color: var(--bs-secondary-color) !important;
            box-shadow: inset 0 0 0 1px var(--bs-secondary-color);
        }

        @media (max-width: 991.98px) {
            :root {
                --dashboard-panel-height: auto;
                --dashboard-chat-height: min(60dvh, 32rem);
            }

            .dashboard-run-info {
                max-height: 20rem;
            }

            .dashboard-step-row.cols-2,
            .dashboard-step-row.cols-3,
            .dashboard-step-row.cols-4,
            .dashboard-step-detail-grid {
                grid-template-columns: 1fr;
            }
        }
        """),
    ui.tags.script("""
        function dashboardInputContainer(input) {
            return input.closest(".form-group, .shiny-input-container") || input.parentElement;
        }

        function setDashboardSelectDisabled(id, disabled) {
            const input = document.getElementById(id);
            if (!input) return;
            const container = dashboardInputContainer(input);
            input.disabled = disabled;
            input.setAttribute("aria-disabled", disabled ? "true" : "false");
            if (container) {
                container.classList.toggle("dashboard-workflow-disabled", disabled);
            }
            if (input.selectize) {
                if (disabled) {
                    input.selectize.disable();
                } else {
                    input.selectize.enable();
                }
            }
        }

        function setDashboardControlDisabled(id, disabled) {
            const input = document.getElementById(id);
            if (!input) return;
            const container = dashboardInputContainer(input);
            input.disabled = disabled;
            input.setAttribute("aria-disabled", disabled ? "true" : "false");
            if (container) {
                container.classList.toggle("dashboard-workflow-disabled", disabled);
            }
            if (input.selectize) {
                if (disabled) {
                    input.selectize.disable();
                } else {
                    input.selectize.enable();
                }
            }
        }

        function dashboardChatTabActive(chat) {
            const tabPane = chat.closest(".tab-pane");
            if (tabPane) return tabPane.classList.contains("active");
            return Boolean(chat.offsetParent);
        }

        function setDashboardChatStreamAutoScroll(stream, enabled) {
            if (enabled) {
                if (!stream.hasAttribute("auto-scroll")) {
                    stream.setAttribute("auto-scroll", "");
                }
            } else {
                if (stream.hasAttribute("auto-scroll")) {
                    stream.removeAttribute("auto-scroll");
                }
            }
        }

        function syncDashboardChatStreamAutoScroll() {
            const chat = document.getElementById("chat");
            if (!chat) return;
            const enabled = dashboardChatTabActive(chat);
            chat.querySelectorAll("shiny-chat-message shiny-markdown-stream").forEach(function(stream) {
                setDashboardChatStreamAutoScroll(stream, enabled);
            });
        }

        function initDashboardChatStreamAutoScrollGuard() {
            const chat = document.getElementById("chat");
            if (!chat || chat.dataset.dashboardAutoScrollGuard === "true") return;
            chat.dataset.dashboardAutoScrollGuard = "true";

            const syncSoon = function() {
                window.requestAnimationFrame(syncDashboardChatStreamAutoScroll);
            };

            const chatObserver = new MutationObserver(syncSoon);
            chatObserver.observe(chat, {
                childList: true,
                subtree: true,
                attributes: true,
                attributeFilter: ["auto-scroll"],
            });

            const tabPane = chat.closest(".tab-pane");
            if (tabPane) {
                const tabObserver = new MutationObserver(syncSoon);
                tabObserver.observe(tabPane, {
                    attributes: true,
                    attributeFilter: ["class"],
                });
            }

            document.addEventListener("shown.bs.tab", syncSoon);
            document.addEventListener("hidden.bs.tab", syncSoon);
            syncSoon();
        }

        if (document.readyState === "loading") {
            document.addEventListener("DOMContentLoaded", initDashboardChatStreamAutoScrollGuard);
        } else {
            initDashboardChatStreamAutoScrollGuard();
        }

        Shiny.addCustomMessageHandler("workflowDispatchState", function(message) {
            const active = Boolean(message && message.active);
            const searchProviderEnabled = Boolean(
                message && message.searchProviderEnabled
            );
            setDashboardSelectDisabled("retrievalEndpoint", active);
            setDashboardSelectDisabled(
                "searchProvider",
                active && !searchProviderEnabled
            );
        });

        Shiny.addCustomMessageHandler("workflowRunControlState", function(message) {
            const disabled = Boolean(message && message.disabled);
            ["advanceWorkflowRun", "runWorkflowToCompletion", "workflowRetryStepID", "retryWorkflowStep"].forEach(function(id) {
                setDashboardControlDisabled(id, disabled);
            });
        });

        document.addEventListener("click", function(event) {
            const button = event.target.closest(".dashboard-step-button");
            if (!button) return;
            const picker = button.closest(".dashboard-step-picker");
            if (!picker) return;

            const panelId = button.getAttribute("data-step-panel");
            const wasOpen = button.getAttribute("aria-expanded") === "true";

            picker.querySelectorAll(".dashboard-step-button").forEach(function(item) {
                item.classList.remove("is-active");
                item.setAttribute("aria-expanded", "false");
            });
            picker.querySelectorAll(".dashboard-step-panel").forEach(function(panel) {
                panel.hidden = true;
            });

            if (wasOpen) return;

            button.classList.add("is-active");
            button.setAttribute("aria-expanded", "true");
            picker.querySelectorAll(".dashboard-step-panel").forEach(function(panel) {
                if (panel.getAttribute("data-step-panel-id") === panelId) {
                    panel.hidden = false;
                }
            });
        });
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
                                "conversationID",
                                "Conversation ID",
                                placeholder="Conversation ID",
                                width="100%",
                            ),
                            "generateConversationID",
                            "🔄",
                            button_class="dashboard-square-button",
                        ),
                        input_action_row(
                            ui.input_select(
                                "retrievalEndpoint",
                                "Retrieval Endpoint",
                                choices=[],
                                width="100%",
                            ),
                            "refreshRetrievalEndpoints",
                            "🔄",
                            button_class="dashboard-square-button",
                        ),
                        ui.input_select(
                            "searchProvider",
                            "Search Provider",
                            choices=[],
                            width="100%",
                        ),
                        ui.input_select(
                            "workflowDispatch",
                            "Workflow Dispatch",
                            choices={"": "None"},
                            selected="",
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
            value="single_node",
        ),
        ui.nav_panel(
            "Multi-Node",
            ui.div("Multi-node UI coming soon!"),
            value="multi_node",
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
                        "workflowConversationID",
                        "Conversation ID",
                        placeholder="Optional",
                        width="100%",
                    ),
                    ui.input_select(
                        "workflowRetrievalEndpoint",
                        "Retrieval Endpoint",
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
                        class_="dashboard-full-width-action",
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
                            ui.input_select(
                                "workflowRetryStepID",
                                "",
                                choices={"": "Select step"},
                                selected="",
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
                col_widths=[3, 9],
            ),
            value="workflows",
        ),
        ui.nav_panel(
            "Conversation History",
            input_action_row(
                ui.input_select(
                    "historyConversationSelector",
                    "Conversation",
                    choices={},
                    width="100%",
                ),
                "refreshHistory",
                "Refresh",
                col_widths=[3, 2],
            ),
            ui.output_ui("historyBox"),
            value="history",
        ),
        ui.nav_panel(
            "Traces",
            ui.layout_columns(
                ui.input_text(
                    "traceConversationFilter",
                    "Conversation ID",
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
            value="traces",
        ),
        id="dashboardNav",
        selected="single_node",
        title="LLM Control Plane",
    ),
    theme=shinyswatch.theme.flatly,
)
