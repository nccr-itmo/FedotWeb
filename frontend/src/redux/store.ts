import { applyMiddleware, combineReducers, createStore } from "redux";
import thunkMiddleware from "redux-thunk";
import { composeWithDevTools } from "redux-devtools-extension";

// redux-persist---
import { persistStore, persistReducer } from "redux-persist";
// import storage from "redux-persist/lib/storage"; // defaults to localStorage for web
import storage from "redux-persist/lib/storage/session"; // sessionStorage

//------
import authReducer from "./reducers/auth/authReducer";
import leftMenuReducer from "./reducers/leftMenu/leftMenuReducer";
import casesReducer from "./reducers/showCase/showCaseReducer";
import sandbox_Egor from "./reducers/sandBox/sandbox-reducer";
import sandBoxReducer from "./reducers/sandBox/sandBoxReducer";
import historyReducer from "./reducers/history/historyReducer";

const persistConfig = {
  key: "root",
  storage,
  whitelist: ["auth", "leftMenu"], //массив для редьюсеров, которые не должны сбрасываться при обновлении
};

let reducers = combineReducers({
  auth: authReducer,
  leftMenu: leftMenuReducer,
  showCases: casesReducer,
  sandbox_Egor,
  sandBox: sandBoxReducer,
  history: historyReducer,
});

type RootReducerType = typeof reducers;
export type StateType = ReturnType<RootReducerType>;

const persistedReducer = persistReducer(persistConfig, reducers); // redux-persist--- сохраняет состояние при обновлении

const store = () => {
  let store = createStore(
    persistedReducer,
    composeWithDevTools(applyMiddleware(thunkMiddleware))
  );
  //@ts-ignore
  let persistor = persistStore(store);
  return { store, persistor };
};

export default store;
