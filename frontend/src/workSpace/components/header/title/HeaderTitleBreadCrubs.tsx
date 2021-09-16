import React, { FC, memo } from "react";
import { useDispatch } from "react-redux";
import scss from "./headerTitle.module.scss";

import Breadcrumbs from "@material-ui/core/Breadcrumbs";
import Link from "@material-ui/core/Link";

import { setHistoryToggle } from "./../../../../redux/reducers/sandBox/sandBoxReducer";

interface I {
  linksAndTitles: { title: string; link: string }[];
  activePage: string;
}

const HeaderTitleBreadCrubs: FC<I> = ({ linksAndTitles, activePage }) => {
  const dispatch = useDispatch();

  function handleClick(event: React.MouseEvent<HTMLDivElement, MouseEvent>) {
    event.preventDefault();
    dispatch(setHistoryToggle(false));
  }

  return (
    <div role="presentation" onClick={handleClick}>
      <Breadcrumbs aria-label="breadcrumb">
        {linksAndTitles.map((link) => (
          <Link
            key={link.link}
            underline="hover"
            color="inherit"
            href={`${link.link}`}
          >
            <p className={scss.text}>{link.title}</p>
          </Link>
        ))}

        <p className={scss.text}>{activePage}</p>
      </Breadcrumbs>
    </div>
  );
};

export default memo(HeaderTitleBreadCrubs);
