import React, { FC, memo, ReactNode } from "react";

import { makeStyles, createStyles } from "@material-ui/core/styles";
import { Button } from "@material-ui/core";

const useStyles = makeStyles(() =>
  createStyles({
    button: {
      marginLeft: 20,
      padding: "11px 4px 12px 0",
      width: 208,

      textTransform: "none",
      borderRadius: "4px",
      background: "#263238",
      "&:hover": {
        background: "#515B5F",
      },
    },
    text: {
      fontFamily: "Open Sans",
      fontSize: "14px",
      lineHeight: "150%",
      letterSpacing: "0.15px",

        color: "#FFFFFF",
      },
    })
);

interface I {
  children: ReactNode;
  onHandleClick(): void;
}

const SandBoxChartBtn: FC<I> = ({ children, onHandleClick }) => {
  const classes = useStyles();

  return (
    <Button className={classes.button} onClick={onHandleClick}>
      <p className={classes.text}>{children}</p>
    </Button>
  );
};

export default memo(SandBoxChartBtn);
