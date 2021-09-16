import React, { memo } from "react";

import { makeStyles, createStyles } from "@material-ui/core/styles";
import MenuIcon from "@material-ui/icons/Menu";

const useStyles = makeStyles(() =>
  createStyles({
    root: {
      marginTop: 26,
    },
    icon: {
      width: 20,
      height: 20,

      color: "#FFFFFF",
    },
  })
);

const LeftMenuMenuIcon = () => {
  const classes = useStyles();

  return (
    <div className={classes.root}>
      <MenuIcon className={classes.icon} />
    </div>
  );
};

export default memo(LeftMenuMenuIcon);
