module Tokenizer : sig
  type t =
    { re : Str.regexp
    ; encoder : (string, int) Hashtbl.t
    ; decoder : (int, string) Hashtbl.t
    ; bpe_ranks : (string * string, int) Hashtbl.t
    ; start_of_text_token : int
    ; end_of_text_token : int
    }

  val make : string -> t
end
