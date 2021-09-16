import React, { FC, memo } from "react";
import scss from "./buttonSiginInUpFormik.module.scss";

import { createStyles, makeStyles } from "@material-ui/core/styles";
import { Button } from "@material-ui/core";

const useStyles = makeStyles(() =>
  createStyles({
    root: {},
    cancelButton: {
      padding: "0 20px",

      textTransform: "none",
      borderRadius: 4,
      background: "#F4F5F6",
    },
    submitButton: {
      padding: "0 16px",

      textTransform: "none",
      borderRadius: 4,
      background: "#828282",

      "&:disabled": {
        background: "#ECEFF1",
      },
      "&:hover": {
        background: "#515B5F",
      },
    },
  })
);

interface I {
  disabled: boolean;
  onCancelClick(): void;
}

const ButtonsCancelAndSubmit: FC<I> = ({ disabled, onCancelClick }) => {
  const classes = useStyles();

  return (
    <div className={scss.buttonsCancelAndSubmit}>
      <Button className={classes.cancelButton} onClick={onCancelClick}>
        <p className={scss.cancelText}>Cancel</p>
      </Button>
      <Button
        type="submit"
        disabled={disabled}
        className={classes.submitButton}
      >
        <p className={scss.submitText}>Submit</p>
      </Button>
    </div>
  );
};

export default memo(ButtonsCancelAndSubmit);
