import React, { FC, memo } from "react";
import scss from "./textFieldSignInFormik.module.scss";

import { createStyles, makeStyles } from "@material-ui/core/styles";
import { TextField } from "@material-ui/core";

import { OnChangeFormikType } from "../onChangeFormikType";

const useStyles = makeStyles(() =>
  createStyles({
    cssOutlinedInput: {
      "&:not(hover):not($disabled):not($cssFocused):not($error) $notchedOutline":
        {
          borderColor: "#90A4AE", //default
        },
      "&:hover:not($disabled):not($cssFocused):not($error) $notchedOutline": {
        borderColor: "rgba(0, 0, 0, 0.7)", //hovered
      },
      "&$cssFocused $notchedOutline": {
        borderColor: "#263238", //focused
      },
    },
    notchedOutline: {},
    cssFocused: {},
    error: {},
    disabled: {},
  })
);

interface ITextFieldSignIn {
  value: string;
  name: string;
  type: "password" | "text";
  label: string;
  error: boolean;
  errorText: string;
  onChange: OnChangeFormikType;
}

const TextFieldSignInFormik: FC<ITextFieldSignIn> = ({
  value,
  name,
  type,
  label,
  error,
  errorText,
  onChange,
}) => {
  const classes = useStyles();

  return (
    <TextField
      name={name}
      value={value}
      onChange={onChange}
      className={scss.root}
      error={error}
      helperText={errorText === "" ? " " : errorText}
      label={label}
      type={type}
      required
      variant="outlined"
      InputLabelProps={{
        classes: {
          root: scss.cssLabel,
          focused: scss.cssFocused,
        },
      }}
      InputProps={{
        classes: {
          root: classes.cssOutlinedInput,
          focused: classes.cssFocused,
          notchedOutline: classes.notchedOutline,
        },
      }}
    />
  );
};

export default memo(TextFieldSignInFormik);
