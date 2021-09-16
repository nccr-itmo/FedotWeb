import React, { memo, useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";

import { makeStyles, createStyles } from "@material-ui/core/styles";
import { CircularProgress, Paper } from "@material-ui/core";

import SandboxPageChartsMetricLine from "./line/SandboxPageChartsMetricLine";
import { getMetrics } from "./../../../../../redux/reducers/sandBox/sandBoxReducer";
import { StateType } from "../../../../../redux/store";

const useStyles = makeStyles(() =>
  createStyles({
    paper: {
      padding: "16px 20px",
    },
    title: {
      fontSize: "18px",
      lineHeight: "150%",
      letterSpacing: "0.15px",

      color: "#263238",
    },
    progress: {
      height: 350,

      display: "flex",
      alignItems: "center",
      justifyContent: "center",
    },
  })
);

const SandboxPageChartsMetric = () => {
  const classes = useStyles();
  const dispatch = useDispatch();

  const { show_case_by_id } = useSelector(
    (state: StateType) => state.showCases
  );
  const { sandbox_metrics } = useSelector((state: StateType) => state.sandBox);

  useEffect(() => {
    if (show_case_by_id) dispatch(getMetrics(show_case_by_id.case_id));
  }, [dispatch, show_case_by_id]);

  return sandbox_metrics ? (
    <Paper className={classes.paper} elevation={3}>
      <p className={classes.title}>Metric</p>
      <SandboxPageChartsMetricLine data={sandbox_metrics} />
    </Paper>
  ) : (
    <div className={classes.progress}>
      <CircularProgress />
    </div>
  );
};

export default memo(SandboxPageChartsMetric);
