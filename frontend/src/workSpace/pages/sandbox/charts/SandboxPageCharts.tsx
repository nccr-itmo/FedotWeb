import React, { memo } from "react";
import scss from "./sandboxPageCharts.module.scss";

import { Grid } from "@material-ui/core";

import SandboxPageChartsBtns from "./buttons/SandboxPageChartsBtns";
import SandboxPageChartsMetric from "./metric/SandboxPageChartsMetric";
import SandboxPageChartsResult from "./result/SandboxPageChartsResult";

const SandboxPageCharts = () => {
  return (
    <div className={scss.root}>
      <SandboxPageChartsBtns />
      <div className={scss.grid}>
        <Grid container spacing={2}>
          <Grid item xs={6}>
            <SandboxPageChartsResult />
          </Grid>
          <Grid item xs={6}>
            <SandboxPageChartsMetric />
          </Grid>
        </Grid>
      </div>
    </div>
  );
};

export default memo(SandboxPageCharts);
