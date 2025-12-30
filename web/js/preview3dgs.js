import { app } from "../../../scripts/app.js";

/**
 * Preview 3D Gaussian Splat - WebGL viewer in iframe
 * Adapts to node size dynamically
 */

app.registerExtension({
    name: "SyntaxNodes.Preview3DGaussianSplat",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "Preview3DGaussianSplat") {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            if (onNodeCreated) {
                onNodeCreated.apply(this, arguments);
            }
            this.size = [400, 400];
        };

        // Handle node resize to update viewer dimensions
        const onResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function(size) {
            if (onResize) {
                onResize.apply(this, arguments);
            }

            // Update iframe container height when node resizes
            const iframeWidget = this.widgets?.find(w => w.name === "gsplat_viewer");
            if (iframeWidget && iframeWidget.container) {
                // Calculate available height (node height minus header and other widgets)
                const headerHeight = 30;
                const inputHeight = 26; // ply_path input
                const padding = 20;
                const availableHeight = Math.max(200, size[1] - headerHeight - inputHeight - padding);
                iframeWidget.container.style.height = `${availableHeight}px`;

                // Notify iframe of resize
                if (iframeWidget.iframeElement && iframeWidget.iframeElement.contentWindow) {
                    iframeWidget.iframeElement.contentWindow.postMessage({ type: 'resize' }, '*');
                }
            }
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            if (onExecuted) {
                onExecuted.apply(this, arguments);
            }

            if (message?.gsplat && message.gsplat.length > 0) {
                const gsplatInfo = message.gsplat[0];

                let iframeWidget = this.widgets?.find(w => w.name === "gsplat_viewer");

                if (!iframeWidget) {
                    const container = document.createElement("div");
                    container.style.cssText = "width:100%;background:#1a1a1a;border-radius:4px;overflow:hidden;";

                    // Calculate initial height based on current node size
                    const headerHeight = 30;
                    const inputHeight = 26;
                    const padding = 20;
                    const initialHeight = Math.max(200, this.size[1] - headerHeight - inputHeight - padding);
                    container.style.height = `${initialHeight}px`;

                    const iframe = document.createElement("iframe");
                    iframe.style.cssText = "width:100%;height:100%;border:none;";
                    iframe.setAttribute("sandbox", "allow-scripts allow-same-origin");
                    container.appendChild(iframe);

                    iframeWidget = this.addDOMWidget("gsplat_viewer", "preview", container, {
                        serialize: false,
                        computeSize: (width) => {
                            // Return minimum height, actual height is set via container style
                            const headerHeight = 30;
                            const inputHeight = 26;
                            const padding = 20;
                            const minHeight = Math.max(200, this.size[1] - headerHeight - inputHeight - padding);
                            return [width, minHeight];
                        }
                    });
                    iframeWidget.iframeElement = iframe;
                    iframeWidget.container = container;
                }

                if (iframeWidget.iframeElement && gsplatInfo.url) {
                    iframeWidget.iframeElement.src = gsplatInfo.url;
                }
            }
        };
    },
});

/**
 * Preview Gaussian Splat Video
 * Uses ComfyUI's native animated image/video preview format
 * Adapts to node size dynamically
 */
app.registerExtension({
    name: "SyntaxNodes.PreviewGaussianSplatVideo",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "PreviewGaussianSplatVideo") {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            if (onNodeCreated) {
                onNodeCreated.apply(this, arguments);
            }
            this.size = [400, 350];
        };

        // Handle node resize to update video dimensions
        const onResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function(size) {
            if (onResize) {
                onResize.apply(this, arguments);
            }

            const videoWidget = this.widgets?.find(w => w.name === "video_preview");
            if (videoWidget && videoWidget.container) {
                const headerHeight = 30;
                const inputHeight = 26;
                const padding = 20;
                const availableHeight = Math.max(150, size[1] - headerHeight - inputHeight - padding);
                videoWidget.container.style.height = `${availableHeight}px`;
            }
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            if (onExecuted) {
                onExecuted.apply(this, arguments);
            }

            // ComfyUI uses {"images": [...], "animated": (true,)} for video preview
            if (message?.images && message.images.length > 0) {
                const videoInfo = message.images[0];

                let videoWidget = this.widgets?.find(w => w.name === "video_preview");

                if (!videoWidget) {
                    const container = document.createElement("div");
                    container.style.cssText = "width:100%;background:#1a1a1a;border-radius:4px;overflow:hidden;";

                    // Calculate initial height based on current node size
                    const headerHeight = 30;
                    const inputHeight = 26;
                    const padding = 20;
                    const initialHeight = Math.max(150, this.size[1] - headerHeight - inputHeight - padding);
                    container.style.height = `${initialHeight}px`;

                    const video = document.createElement("video");
                    video.style.cssText = "width:100%;height:100%;object-fit:contain;";
                    video.controls = true;
                    video.loop = true;
                    video.muted = true;
                    video.autoplay = true;
                    container.appendChild(video);

                    videoWidget = this.addDOMWidget("video_preview", "preview", container, {
                        serialize: false,
                        computeSize: (width) => {
                            const headerHeight = 30;
                            const inputHeight = 26;
                            const padding = 20;
                            const minHeight = Math.max(150, this.size[1] - headerHeight - inputHeight - padding);
                            return [width, minHeight];
                        }
                    });
                    videoWidget.videoElement = video;
                    videoWidget.container = container;
                }

                const params = new URLSearchParams({
                    filename: videoInfo.filename,
                    subfolder: videoInfo.subfolder || "",
                    type: videoInfo.type || "output",
                });
                const videoUrl = `/view?${params.toString()}`;

                if (videoWidget.videoElement) {
                    videoWidget.videoElement.src = videoUrl;
                    videoWidget.videoElement.load();
                }
            }
        };
    },
});
