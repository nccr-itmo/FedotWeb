import React, { memo } from "react";
import { useDispatch, useSelector } from "react-redux";

import { Grid } from "@material-ui/core";
import { makeStyles, createStyles } from "@material-ui/core/styles";

import { StateType } from "../../../../../redux/store";
import { setgenothypicPhenothypic } from "../../../../../redux/reducers/history/historyReducer";

const useStyles = makeStyles(() =>
  createStyles({
    item: {
      textAlign: "center",
    },
    text: {
      fontFamily: "Open Sans",
      fontWeight: "normal",
      fontSize: "18px",
      lineHeight: "120%",
      letterSpacing: "0.15px",

      color: "#263238",

      "&:hover": {
        background: "#E2E7EA",
        cursor: "pointer",

        borderRadius: 8,

                transition: "all 0.5s",
            },
        },
    textActive: {
      fontFamily: "Open Sans",
      fontWeight: "bold",
      fontSize: "18px",
      lineHeight: "120%",
      letterSpacing: "0.15px",

      color: "#263238",
    },
  })
);

const HistoryPageChartHeader = () => {
  const classes = useStyles();
  const dispatch = useDispatch();

  const { history_genothypic_phenothypic } = useSelector(
    (state: StateType) => state.history
  );

  const onPhenothypeClick = () => {
    dispatch(setgenothypicPhenothypic(true));
  };
  const onGenotypicClick = () => {
    dispatch(setgenothypicPhenothypic(false));
  };

  return (
    <Grid container spacing={0}>
      <Grid item xs={6} className={classes.item}>
        <p
          onClick={onPhenothypeClick}
          className={
            history_genothypic_phenothypic ? classes.textActive : classes.text
          }
        >
          Phenothypic convergence
        </p>
      </Grid>
      <Grid item xs={6} className={classes.item}>
        <p
          onClick={onGenotypicClick}
          className={
            history_genothypic_phenothypic ? classes.text : classes.textActive
          }
        >
          Genotypic convergence
        </p>
      </Grid>
    </Grid>
  );
};

export default memo(HistoryPageChartHeader);
