import * as d3 from "d3";
import "./HistoryGraph.module.scss";
import dagreD3 from "dagre-d3";
import styles from "./HistoryGraph.module.scss";

export function runHistory(
  container,
  linksData,
  nodesData,
  nodeHoverTooltip,
  onClick
) {

  let links = linksData.map((d) => Object.assign({}, d));
  let nodes = nodesData.map((d) => Object.assign({}, d));
  let nodesIndividual = nodes.filter((item) => item.type === "individual");
  let nodesOperator = nodes.filter((item) => item.type === "evo_operator");
  let generations = [];
  nodesIndividual.forEach((item) => generations.push(item.gen_id));

 
  // оставляем уникальные значения
  let uniqGenerations = Array.from(new Set(generations));

  const containerRect = container.getBoundingClientRect();
  const height = containerRect.height;
  const width = containerRect.width;

  const svg = d3
    .select(container)
    .append("svg")
    .attr("id", "graphSvg")
    .attr("viewBox", [-width / 2, -height / 2, width, height])
    .call(
      d3.zoom().on("zoom", function (event) {
        svgGroup.attr("transform", event.transform);
      })
    );

  // Set up an SVG group so that we can translate the final graph.
  let svgGroup = svg.append("g").on("click", (e, d) => {   
    if (e.target && e.target.textContent.includes("pipeline")) {
      nodesIndividual.find(item=>item.uid === e.target.textContent)
      // onClick(e.target.textContent);
      onClick(nodesIndividual.find(item=>item.uid === e.target.textContent));
    }
  });

  // Create the input graph
  let g = new dagreD3.graphlib.Graph({ compound: true })
    .setGraph({
      rankdir: "TB",
      ranker: "network-simplex",
      align: "DL",
    })
    .setDefaultEdgeLabel(() => ({}));

  // Here we're setting nodeclass, which is used by our custom drawNodes function
  // below.

  //Кластеры поколений

  uniqGenerations.forEach((generation) => {
    let nameG = `Generation ${generation}`;

    g.setNode(generation, {
      label: generation,
      clusterLabelPos: "left",
      style: "fill: #d3d7e8",
    });

    //узел отображающий поколение
    g.setNode(nameG, {
      label: nameG,
    });
    g.setParent(nameG, generation);

    nodesIndividual.forEach((item) => {
      // console.log(`item`, item)
      if (generation === item.gen_id) {
        g.setNode(item.uid, {
          label: item.uid,   
          labelStyle: "font-size: 36px; cursor: pointer",
          class: "type-TK",
          shape: "rect",
        });

        g.setParent(item.uid, generation);
      }
    });
  });

  nodesOperator.forEach((item) => {
    const label = item.name[0][0].toUpperCase();
    g.setNode(item.uid, {
      label: label, 
      labelStyle: "font-size: 60px; cursor: pointer",
      class: `operator-${label}`,
      shape: "circle",
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
        style: "stroke: #007DFF; stroke-width: 4px; fill-opacity: 0",
        arrowheadStyle: "fill: #007DFF",
        curve: d3.curveBasis,
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

  /**
   * Блок добавления и удаления подсказки на Граф
   */
  // const tooltip = document.querySelector("#graph-tooltip");
  // if (!tooltip) {
  //   const tooltipDiv = document.createElement("div");
  //   tooltipDiv.classList.add(styles.tooltip);
  //   tooltipDiv.style.opacity = "0";
  //   tooltipDiv.id = "graph-tooltip";
  //   document.body.appendChild(tooltipDiv);
  // }
  // const div = d3.select("#graph-tooltip");

  // //показать рядом с нужным Узлом
  // const addTooltip = (hoverTooltip, d, x, y) => {
  //   div.transition().duration(200).style("opacity", 0.9);
  //   div
  //     .html(hoverTooltip(d))
  //     .style("left", `${x}px`)
  //     .style("top", `${y - 28}px`);
  // };
  // //скрыть
  // const removeTooltip = () => {
  //   div.transition().duration(200).style("opacity", 0);
  // };
  /**
   * Конец блока добавления и удаления Подсказки
   */

  svgGroup
    .selectAll("g.node")
    // .on("mouseover", (event, d) => {
    //   addTooltip(nodeHoverTooltip, d, event.pageX, event.pageY);
    // })
    // .on("mouseout", () => {
    //   removeTooltip();
    // });

  return {
    destroy: () => {
      return svg.remove();
    },
    nodes: () => {
      return svg.node();
    },
  };
}
