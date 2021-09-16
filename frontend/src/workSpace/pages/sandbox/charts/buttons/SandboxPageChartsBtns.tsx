import React, { memo, useCallback } from "react";
import { useDispatch } from "react-redux";

import { makeStyles, createStyles } from "@material-ui/core/styles";
import { Grid } from "@material-ui/core";

import SandBoxChartBtn from "../../../../../ui/buttons/SandBoxChartBtn";
import { setHistoryToggle } from "../../../../../redux/reducers/sandBox/sandBoxReducer";

const useStyles = makeStyles(() =>
  createStyles({
    btns: { display: "flex", alignItems: "center", justifyContent: "flex-end" },
  })
);

const SandboxPageChartsBtns = () => {
  const classes = useStyles();
  const dispatch = useDispatch();

  const onHistoryClick = useCallback(() => {
    dispatch(setHistoryToggle(true));
  }, [dispatch]);

  const onExportClick = useCallback(() => {}, []);

  return (
    <Grid container spacing={1}>
      <Grid item xs={12} className={classes.btns}>
        {/*<SandBoxChartBtn onHandleClick={onExportClick}>
          Export Model
        </SandBoxChartBtn>*/}
        <SandBoxChartBtn onHandleClick={onHistoryClick}>
          History
        </SandBoxChartBtn>
      </Grid>
    </Grid>
  );
};

export default memo(SandboxPageChartsBtns);
