import * as d3 from "d3";
import "./directedGraph.module.scss";
import dagreD3 from "dagre-d3";


export function runDirectedGraph(
  container,
  linksData,
  nodesData,
  onContextMenuNode,
  onContextMenuEdge,
  onMouseDownNode,
  onMouseUpNode,
  onMouseUpSvg
) {
  let links = linksData.map((d) => Object.assign({}, d));
  let nodes = nodesData.map((d) => Object.assign({}, d));



  const containerRect = container.getBoundingClientRect();
  const height = containerRect.height;
  const width = containerRect.width;

  const svg = d3
    .select(container)
    .append("svg")
    .attr("id", "graphSvg")
    .attr("viewBox", [-width / 2, -height / 2, width, height]);
  // .call(
  //   d3.zoom().on("zoom", function (event) {
  //     svgGroup.attr("transform", event.transform);
  //   })
  // );

  // Set up an SVG group so that we can translate the final graph.
  let svgGroup = svg.append("g");

  // Create the input graph
  let g = new dagreD3.graphlib.Graph()
    .setGraph({ rankdir: "LR" })
    .setDefaultEdgeLabel(() => ({}));

  // Here we're setting nodeclass, which is used by our custom drawNodes function
  // below.
  nodes.forEach((item) => {
    g.setNode(item.id , {
  // тут отрисовка элемента графа
 
      label: `${item.display_name} id:${item.id}`,
      shape: item.type === "model" ? "circle" : "rect",
      style: item.type === "model" ? "fill: #00ffd0" : "fill: #B0BEC5",
    
    });
  });

  g.nodes().forEach(function (v) {
    let node = g.node(v);
    // Round the corners of the nodes
    node.rx = node.ry = 5;
  });

  // Set up edges, no special attributes.
  links.forEach((item) => {
    const haveSource = g.hasNode(item.source);
    const haveTarget = g.hasNode(item.target);
    if (haveSource && haveTarget) {
      g.setEdge(item.source, item.target, {
        arrowheadStyle: "fill: gold",
      });
    }
  });

  // Create the renderer
  let render = new dagreD3.render();

  // Run the renderer. This is what draws the final graph.
  render(svgGroup, g);

  // Center the graph
  const gHeight = g.graph().height;
  const gWidth = g.graph().width;
  let xCenterOffset = (svg.attr("width") - gWidth) / 2;
  let yCenterOffset = (svg.attr("height") - gHeight) / 2;

  if (height < gHeight || width < gWidth) {
    const viewHeight = height < gHeight ? gHeight : height;
    const viewWidth = width < gWidth ? gWidth : width;
    svg.attr("viewBox", [
      -viewWidth / 2,
      -viewHeight / 2,
      viewWidth,
      viewHeight,
    ]);
  }
  svgGroup.attr(
    "transform",
    "translate(" + xCenterOffset + "," + yCenterOffset + ")"
  );

  svg
    .selectAll("g.node")
    .on("contextmenu", onContextMenuNode)
  //   .on("mousedown", (e, d) => {
  //     if (e.shiftKey) {
  //       onMouseDownNode({
  //         data: d,
  //         offset: {
  //           x: e.offsetX,
  //           y: e.offsetY,
  //         },
  //       });
  //     }
  //   })
  //   .on("mouseup", (e, d) => {
  //     if (e.shiftKey) {
  //       onMouseUpNode({
  //         data: d,
  //         offset: {
  //           x: e.offsetX,
  //           y: e.offsetY,
  //         },
  //       });
  //     }
  //   });

  // svg.on("mouseup", (e, d) => {
  //   onMouseUpSvg(e, d);
  // });

  svg.selectAll("g.edgePath")
  // .on("contextmenu", onContextMenuEdge);

  return {
    destroy: () => {
      return svg.remove();
    },
    nodes: () => {
      return svg.node();
    },
  };
}
