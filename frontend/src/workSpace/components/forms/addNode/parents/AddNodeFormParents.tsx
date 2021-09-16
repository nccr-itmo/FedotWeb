import React, { FC, memo } from "react";

import { makeStyles, createStyles } from "@material-ui/core/styles";
import { Grid } from "@material-ui/core";

import { OnChangeFormikType } from "../../../../../ui/formik/onChangeFormikType";
import TextFieldFormik from "../../../../../ui/formik/textFields/TextFieldFormik";

const useStyles = makeStyles(() =>
  createStyles({
    root: {
      marginTop: 10,
    },

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
  values: { parents: string };
  onChange: OnChangeFormikType;
}

const AddNodeFormParents: FC<I> = ({ values, onChange }) => {
  const classes = useStyles();

  return (
    <Grid container spacing={0} className={classes.root}>
      <Grid item xs={6} className={`${classes.textContainer} ${classes.item}`}>
        <p className={classes.text}>Parents ids</p>
      </Grid>
      <Grid item xs={1}></Grid>
      <Grid item xs={5}>
        <TextFieldFormik
          value={values.parents}
          onChange={onChange}
          name="parents"
        />
      </Grid>
    </Grid>
  );
};

export default memo(AddNodeFormParents);
