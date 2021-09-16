import React, { FC, memo } from "react";

import { makeStyles, createStyles } from "@material-ui/core/styles";
import { Grid } from "@material-ui/core";

import TextFieldFormik from "../../../../../ui/formik/textFields/TextFieldFormik";
import { OnChangeFormikType } from "../../../../../ui/formik/onChangeFormikType";

const useStyles = makeStyles(() =>
  createStyles({
    root: {},

    item: {
      display: "flex",
      alignItems: "center",
    },

    textContainer: {
      borderRadius: 10,
      background: "#828282",
    },
    text: {
      paddingLeft: 9,

            fontFamily: "Open Sans",
            fontSize: "14px",
            lineHeight: "18px",
            letterSpacing: "0.1px",

            color: " #ffffff",
        },
    })
);

interface I {
  values: { name: string };
  onChange: OnChangeFormikType;
}

const AddNodeFormName: FC<I> = ({ values, onChange }) => {
  const classes = useStyles();

  return (
    <Grid container spacing={0}>
      <Grid item xs={4} className={`${classes.textContainer} ${classes.item}`}>
        <p className={classes.text}>Name</p>
      </Grid>
      <Grid item xs={1}></Grid>
      <Grid item xs={7}>
        <TextFieldFormik value={values.name} onChange={onChange} name="name" />
      </Grid>
    </Grid>
  );
};

export default memo(AddNodeFormName);
