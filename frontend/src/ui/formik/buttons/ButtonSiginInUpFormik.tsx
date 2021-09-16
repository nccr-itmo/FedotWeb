import React, { FC, memo, ReactNode } from "react";
import scss from "./buttonSiginInUpFormik.module.scss";

import { createStyles, makeStyles } from "@material-ui/core/styles";
import { Button } from "@material-ui/core";

const useStyles = makeStyles(() =>
  createStyles({
    button: {
      height: 48,
      width: "100%",

      borderRadius: "4px",
      background: "#263238",
      textTransform: "none",
      "&:disabled": {
        background: "#ECEFF1",
      },
      "&:hover": {
        background: "#515B5F",
      },
    },
  })
);

interface IButtonSiginInUpFormik {
  children: ReactNode;
  disabled: boolean;
}

const ButtonSiginInUpFormik: FC<IButtonSiginInUpFormik> = ({
  disabled,
  children,
}) => {
  const classes = useStyles();

  return (
    <Button className={classes.button} type="submit" disabled={disabled}>
      <p className={scss.text}>{children}</p>
    </Button>
  );
};

export default memo(ButtonSiginInUpFormik);
