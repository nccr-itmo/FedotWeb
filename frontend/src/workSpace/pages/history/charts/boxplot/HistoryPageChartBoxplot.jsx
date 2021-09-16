import React, { memo } from "react";
import Plot from "react-plotly.js";
import scss from "../historyPageChart.module.scss";

import { sortByY } from "./../../../../../helpers/sort/sort";
import { CircularProgress } from "@material-ui/core";

const HistoryPageChartBoxplot = ({ data }) => {
  return data ? (
    <Plot
      data={sortByY(data)}
      layout={{
        margin: { t: 20 },
        height: 150 + 40 * data.length,
        width: 370,
        showlegend: false,
        yaxis: { fixedrange: true },
        xaxis: {
          fixedrange: true,
          rangemode: "normal",
        },
      }}
      config={{
        moauseZoom: false,
        displayModeBar: false,
        showlegend: true,
      }}
    />
  ) : (
    <div className={scss.progress}>
      <CircularProgress />
    </div>
  );
};

export default memo(HistoryPageChartBoxplot);
