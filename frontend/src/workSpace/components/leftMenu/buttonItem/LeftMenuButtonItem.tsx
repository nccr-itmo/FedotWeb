import React, { FC, memo, ReactNode } from "react";

import { makeStyles, createStyles } from "@material-ui/core/styles";
import { Button } from "@material-ui/core";

const useStyles = makeStyles(() =>
  createStyles({
    button: {
      padding: "4px",
      width: "100%",

      display: "flex",
      alignItems: "center",

      textTransform: "none",
      borderRadius: 0,

      "&:disabled": {
        background: "#ECEFF1",
      },
      "&:hover": {
        background: "#B0BEC5",
      },
    },
    selected: {
      background: "#4F5B62",
    },
    text: {
      paddingLeft: 15,

      fontFamily: "Open Sans",

      fontSize: "14px",
      lineHeight: "150%",
      letterSpacing: "0.15px",

      transition: "all 1s",
      color: "#FFFFFF",
    },
    textHiden: {
      display: "hidden",

      transition: "all 1s",
    },
  })
);

interface ILeftMenuButtonItem {
  title: "Showcase" | "Sandbox" | "FEDOT" | "Settings";
  buttonTextCss: any;
  buttonWidthCss: any;
  children: ReactNode;
  selected: boolean;
  onClick(page: "Showcase" | "Sandbox" | "FEDOT" | "Settings"): void;
}

const LeftMenuButtonItem: FC<ILeftMenuButtonItem> = ({
  title,
  children,
  buttonTextCss,
  buttonWidthCss,
  selected,
  onClick,
}) => {
  const classes = useStyles();

  return (
    <div className={buttonWidthCss}>
      <Button
        className={
          selected ? `${classes.button} ${classes.selected}` : classes.button
        }
        onClick={() => {
          onClick(title);
        }}
      >
        {children}
        <p className={buttonTextCss}>{title}</p>
      </Button>
    </div>
  );
};

export default memo(LeftMenuButtonItem);
