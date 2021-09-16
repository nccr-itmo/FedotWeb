import React, { memo } from "react";

import { makeStyles } from "@material-ui/core/styles";

import GraphEditor from "./GraphEditor/GraphEditor";
import SandboxPageEpoch from "./epoch/SandboxPageEpoch";
import SandboxPageCharts from "./charts/SandboxPageCharts";

const useStyles = makeStyles(() => ({
  root: {
    display: "flex",
    flexDirection: "column",
  },
}));

const SandboxPage = () => {
  const classes = useStyles();

  return (
    <div className={classes.root}>
      <GraphEditor />
      <SandboxPageEpoch />
      <SandboxPageCharts />
    </div>
  );
};

export default memo(SandboxPage);
