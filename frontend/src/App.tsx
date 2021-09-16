import React from "react";
import { BrowserRouter, Route, Switch } from "react-router-dom";

import "typeface-roboto";
import "typeface-open-sans";

import SignInPage from "./workSpace/pages/signIn/SignInPage";
import WorkSpace from "./workSpace/WorkSpace";

function App() {
  return (
    <BrowserRouter>
      <div className="App">
        <Switch>
          <Route exact path="/" render={() => <SignInPage />} />
          <Route path="/ws/" render={() => <WorkSpace />} />
        </Switch>
      </div>
    </BrowserRouter>
  );
}

export default App;
