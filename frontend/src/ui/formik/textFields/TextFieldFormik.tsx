import React, { FC, memo } from "react";

import { createStyles, makeStyles } from "@material-ui/core/styles";
import { TextField } from "@material-ui/core";

import { OnChangeFormikType } from "../onChangeFormikType";

const useStyles = makeStyles(() =>
  createStyles({
    root: {
      // color: "red",
      "& .MuiOutlinedInput-input": {
        padding: "4px 8px",

        fontFamily: "Open Sans",
        fontSize: "14px",
        lineHeight: "18px",
        letterSpacing: "0.1px",

        color: "#263238",
      },
      "& .MuiOutlinedInput-multiline": {
        padding: 0,
      },
      "& .MuiOutlinedInput-root": {
        minHeight: 24,

          boxSizing: "border-box",
          borderRadius: "10px",

          "&:hover fieldset": {
            borderColor: "#064477",
            transition: "borderColor 1s",
        },

        "&.Mui-focused fieldset": {
          border: "1px solid #064477",
          boxSizing: "border-box",
          borderRadius: "10px",
        },
      },
    },
  })
);

interface ITextFieldFormik {
  value: string | number;
  name: string;
  type?: "string" | "number";
  disabled?: boolean;
  error?: boolean;
  errorText?: string;
  onChange: OnChangeFormikType;
}

const TextFieldFormik: FC<ITextFieldFormik> = ({
  value,
  name,
  type,
  disabled,
  error,
  errorText,
  onChange,
}) => {
  const classes = useStyles();

  return (
    <TextField
      type={type}
      name={name}
      disabled={disabled}
      placeholder={type === "number" ? "0" : ""}
      value={value}
      error={error}
      helperText={error ? errorText : ""}
      onChange={onChange}
      className={classes.root}
      // classes={{
      //   root: classes.root,
      // }}
      variant="outlined"
    />
  );
};

export default memo(TextFieldFormik);
