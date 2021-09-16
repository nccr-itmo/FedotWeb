import React, { memo } from "react";
import scss from "./headerTitle.module.scss";
import { useSelector } from "react-redux";

import GradientIcon from "@material-ui/icons/Gradient";
import { StateType } from "../../../../redux/store";
import HeaderTitleBreadCrubs from "./HeaderTitleBreadCrubs";

const linksHistory = [{ link: "/ws/sandbox/", title: "Sandbox" }];

const HeaderTitle = () => {
  const { page_selected } = useSelector((state: StateType) => state.leftMenu);
  const { sandbox_history } = useSelector((state: StateType) => state.sandBox);

  return (
    <div className={scss.root}>
      <GradientIcon className={scss.icon} />
      <HeaderTitleBreadCrubs
        linksAndTitles={sandbox_history ? linksHistory : []}
        activePage={sandbox_history ? "History" : page_selected}
      />
      {/* <p className={scss.text}></p> */}
    </div>
  );
};

export default memo(HeaderTitle);
