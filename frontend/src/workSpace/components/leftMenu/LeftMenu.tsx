import React, { memo, useCallback } from "react";
import scss from "./leftMenu.module.scss";
import { useDispatch, useSelector } from "react-redux";

import { makeStyles, createStyles } from "@material-ui/core/styles";
import GradientIcon from "@material-ui/icons/Gradient";
import TerrainIcon from "@material-ui/icons/Terrain";
import AccountTreeIcon from "@material-ui/icons/AccountTree";
import SettingsIcon from "@material-ui/icons/Settings";

import LeftMenuMenuIcon from "./menuIcon/LeftMenuMenuIcon";
import LogoMenu from "../../../data/images/LogoMenu.png";
import LeftMenuButtonItem from "./buttonItem/LeftMenuButtonItem";
import { selectPage } from "../../../redux/reducers/leftMenu/leftMenuReducer";
import { StateType } from "../../../redux/store";
import { setHistoryToggle } from "./../../../redux/reducers/sandBox/sandBoxReducer";
import { getShowCaseById } from "../../../redux/reducers/showCase/showCaseReducer";

const useStyles = makeStyles(() =>
  createStyles({
    icon: {
      width: 24,
      height: 24,

      color: "#FFFFFF",
    },
  })
);

const LeftMenu = () => {
  const classes = useStyles();
  const dispatch = useDispatch();
  const { page_selected } = useSelector((state: StateType) => state.leftMenu);

  const btnArr = [
    {
      name: "Showcase" as "Showcase",
      icon: <GradientIcon className={classes.icon} />,
    },
    {
      name: "Sandbox" as "Sandbox",
      icon: <TerrainIcon className={classes.icon} />,
    },
    {
      name: "FEDOT" as "FEDOT",
      icon: <AccountTreeIcon className={classes.icon} />,
    },
    {
      name: "Settings" as "Settings",
      icon: <SettingsIcon className={classes.icon} />,
    },
  ];

  const onMenuClick = useCallback(
    (page: "Showcase" | "Sandbox" | "FEDOT" | "Settings") => {
      dispatch(setHistoryToggle(false));
      dispatch(selectPage(page));

      if (page === "Sandbox") {
        dispatch(getShowCaseById("scoring"));
      }
    },
    [dispatch]
  );

  return (
    <section className={scss.root}>
      <LeftMenuMenuIcon />
      <div className={scss.line} />
      <img src={LogoMenu} alt="Fedot" className={scss.logo} />
      <div className={scss.buttonsPosition}>
        {btnArr.map((btn) =>
          btn.name === "FEDOT" ? (
            <a href={"https://github.com/nccr-itmo/FEDOT"} key={btn.name + "q"}>
              <LeftMenuButtonItem
                title={btn.name}
                selected={page_selected === btn.name}
                buttonTextCss={scss.buttonText}
                buttonWidthCss={scss.buttonWidth}
                onClick={onMenuClick}
              >
                {btn.icon}
              </LeftMenuButtonItem>
            </a>
          ) : (
            <LeftMenuButtonItem
              key={btn.name + "q"}
              title={btn.name}
              selected={page_selected === btn.name}
              buttonTextCss={scss.buttonText}
              buttonWidthCss={scss.buttonWidth}
              onClick={onMenuClick}
            >
              {btn.icon}
            </LeftMenuButtonItem>
          )
        )}
      </div>
      <div className={scss.line} />
    </section>
  );
};

export default memo(LeftMenu);
