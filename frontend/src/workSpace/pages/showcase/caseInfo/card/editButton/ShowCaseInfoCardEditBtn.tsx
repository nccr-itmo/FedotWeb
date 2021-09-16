import React from "react";

import { makeStyles, createStyles } from "@material-ui/core/styles";
import { Button, Grid } from "@material-ui/core";
import { useDispatch } from "react-redux";
import { selectPage } from "../../../../../../redux/reducers/leftMenu/leftMenuReducer";
import { setHistoryToggle } from "../../../../../../redux/reducers/sandBox/sandBoxReducer";

const useStyles = makeStyles(() =>
  createStyles({
    container: {
      marginTop: 8,
    },
    item: {
      display: "flex",
      alignItems: "center",
      justifyContent: "flex-end",
    },
    btn: {
      textTransform: "none",

      borderRadius: 4,

      background: "#F5F5F6",
      "&:hover": {
        background: "#90A4AE",
      },
    },
    text: {
      fontWeight: 300,
      fontSize: "14px",
      lineHeight: "24px",
      letterSpacing: "0.1px",

      color: "#000000",
    },
  })
);

const ShowCaseInfoCardEditBtn = () => {
  const classes = useStyles();
  const dispatch = useDispatch();

  const onEditClick = () => {
    dispatch(setHistoryToggle(false));
    dispatch(selectPage("Sandbox"));
  };

  return (
    <Grid container className={classes.container}>
      <Grid item xs={12} className={classes.item}>
        <Button onClick={onEditClick} className={classes.btn}>
          <p className={classes.text}>Edit in Sandbox</p>
        </Button>
      </Grid>
    </Grid>
  );
};

export default ShowCaseInfoCardEditBtn;
