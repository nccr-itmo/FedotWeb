import React, { FC, memo, ReactNode } from "react";

import { makeStyles, createStyles } from "@material-ui/core/styles";
import { Button } from "@material-ui/core";

const useStyles = makeStyles(() =>
  createStyles({
    button: {
      padding: 4,
      textTransform: "none",

      background: "#1A2327",
      "&:hover": {
        background: "#515B5F",
      },
    },
    text: {
      fontFamily: "Open Sans",
      fontSize: "14px",
      lineHeight: "24px",
      letterSpacing: "0.1px",

            color: "#FFFFFF",
        },
    })
);

interface ICardButton {
  id: string;
  children: ReactNode;
  onClick(value: string): void;
}

const CardButton: FC<ICardButton> = ({ id, children, onClick }) => {
  const classes = useStyles();

  return (
    <Button className={classes.button} onClick={(e) => onClick(id)}>
      <p className={classes.text}>{children}</p>
    </Button>
  );
};

export default memo(CardButton);
