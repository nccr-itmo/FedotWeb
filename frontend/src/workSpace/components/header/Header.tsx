import React, { memo } from "react";
import scss from "./header.module.scss";

import { makeStyles, createStyles } from "@material-ui/core/styles";
import { Button } from "@material-ui/core";
import AccountCircleIcon from "@material-ui/icons/AccountCircle";

import HeaderTitle from "./title/HeaderTitle";
import AddButton from "../../../ui/buttons/AddButton";

const useStyles = makeStyles(() =>
  createStyles({
    buttonProfile: {
      marginLeft: 20,
      padding: "0 4px",
      minWidth: 0,
    },
    icon: {
      width: 21,
      height: 21,
    },
  })
);

const Header = () => {
  const classes = useStyles();
  return (
    <section className={scss.root}>
      <HeaderTitle />
      <div className={scss.container}>
          {/*<AddButton>Submit New Model</AddButton>*/}
        <div className={scss.line} />
        <Button className={classes.buttonProfile}>
          <AccountCircleIcon className={classes.icon} />
        </Button>
      </div>
    </section>
  );
};

export default memo(Header);
