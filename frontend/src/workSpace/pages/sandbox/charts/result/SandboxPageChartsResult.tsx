import React, { memo, useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";

import { makeStyles, createStyles } from "@material-ui/core/styles";
import { CircularProgress, Paper } from "@material-ui/core";
import { getResults } from "../../../../../redux/reducers/sandBox/sandBoxReducer";
import { StateType } from "../../../../../redux/store";
import SandboxPageChartsResultScatter from "./scatter/SandboxPageChartsResultScatter";
import SandboxPageChartsResultLine from "./line/SandboxPageChartsResultLine";

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

const SandboxPageChartsResult = () => {
  const classes = useStyles();
  const dispatch = useDispatch();

  const { show_case_by_id } = useSelector(
    (state: StateType) => state.showCases
  );
  const { sandbox_results, sandBox_epochs_selected } = useSelector(
    (state: StateType) => state.sandBox
  );

  useEffect(() => {
    if (sandBox_epochs_selected && show_case_by_id)
      dispatch(
        getResults(show_case_by_id.case_id, sandBox_epochs_selected.pipeline_id)
      );
    else if (show_case_by_id)
      dispatch(
        getResults(show_case_by_id.case_id, show_case_by_id.pipeline_id)
      );
  }, [dispatch, show_case_by_id, sandBox_epochs_selected]);

  return sandbox_results ? (
    <Paper className={classes.paper} elevation={3}>
      <p className={classes.title}>Modeling Result</p>

      {sandbox_results?.options.chart.type === "line" ? (
        <SandboxPageChartsResultLine data={sandbox_results} />
      ) : sandbox_results?.options.chart.type === "scatter" ? (
        <SandboxPageChartsResultScatter data={sandbox_results} />
      ) : (
        <> </>
      )}
    </Paper>
  ) : (
    <div className={classes.progress}>
      <CircularProgress />
    </div>
  );
};

export default memo(SandboxPageChartsResult);
