import React, { memo, useEffect } from "react";
import scss from "./workSpace.module.scss";
import { Route, Switch, useHistory } from "react-router-dom";
import { useSelector } from "react-redux";

import LeftMenu from "./components/leftMenu/LeftMenu";
import Header from "./components/header/Header";
import SandboxPage from "./pages/sandbox/SandboxPage";
import { StateType } from "../redux/store";
import HistoryPage from "./pages/history/HistoryPage";
import ShowcasePage from "./pages/showcase/ShowCasePage";

const WorkSpace = () => {
  const history = useHistory();
  const { page_selected } = useSelector((state: StateType) => state.leftMenu);
  const { sandbox_history } = useSelector((state: StateType) => state.sandBox);

  useEffect(() => {
    if (page_selected === "Showcase") {
      history.push("/ws/showcase/");
    }
    if (page_selected === "Sandbox") {
      history.push("/ws/sandbox/");
    }

    if (page_selected === "Sandbox" && sandbox_history) {
      history.push("/ws/sandbox/history/");
    }

    if (page_selected === "FEDOT") {
      history.push("/ws/fedot/");
    }
    if (page_selected === "Settings") {
      history.push("/ws/settings/");
    }
  }, [page_selected, history, sandbox_history]);

  return (
    <section className={scss.root}>
      <div className={scss.leftMenu}>
        <LeftMenu />
      </div>
      <div className={scss.header}>
        <Header />
      </div>
      <div className={scss.content}>
        <Switch>
          <Route exact path="/ws/showcase/" render={() => <ShowcasePage />} />
          <Route exact path="/ws/sandbox/" render={() => <SandboxPage />} />
          <Route
              exact
              path="/ws/sandbox/history/"
              render={() => <HistoryPage/>}
          />
          {/* <Route exact path="/ws/fedot/" render={() => <HistoryPage />} /> */}
        </Switch>
      </div>
    </section>
  );
};

export default memo(WorkSpace);
