let _vocab_size = 49408
let _embed_dim = 768
let _intermediate_size = 3072
let _max_position_embeddings = 77
let _num_hidden_layers = 12
let _num_attention_heads = 12

let pat =
  Re2.create_exn
    {|<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+|}
;;

let bytes_to_unicode =
  [| 33, "!"
   ; 34, "\""
   ; 35, "#"
   ; 36, "$"
   ; 37, "%"
   ; 38, "&"
   ; 39, "'"
   ; 40, "("
   ; 41, ")"
   ; 42, "*"
   ; 43, "+"
   ; 44, ","
   ; 45, "-"
   ; 46, "."
   ; 47, "/"
   ; 48, "0"
   ; 49, "1"
   ; 50, "2"
   ; 51, "3"
   ; 52, "4"
   ; 53, "5"
   ; 54, "6"
   ; 55, "7"
   ; 56, "8"
   ; 57, "9"
   ; 58, ":"
   ; 59, ";"
   ; 60, "<"
   ; 61, "="
   ; 62, ">"
   ; 63, "?"
   ; 64, "@"
   ; 65, "A"
   ; 66, "B"
   ; 67, "C"
   ; 68, "D"
   ; 69, "E"
   ; 70, "F"
   ; 71, "G"
   ; 72, "H"
   ; 73, "I"
   ; 74, "J"
   ; 75, "K"
   ; 76, "L"
   ; 77, "M"
   ; 78, "N"
   ; 79, "O"
   ; 80, "P"
   ; 81, "Q"
   ; 82, "R"
   ; 83, "S"
   ; 84, "T"
   ; 85, "U"
   ; 86, "V"
   ; 87, "W"
   ; 88, "X"
   ; 89, "Y"
   ; 90, "Z"
   ; 91, "["
   ; 92, "\\"
   ; 93, "]"
   ; 94, "^"
   ; 95, "_"
   ; 96, "`"
   ; 97, "a"
   ; 98, "b"
   ; 99, "c"
   ; 100, "d"
   ; 101, "e"
   ; 102, "f"
   ; 103, "g"
   ; 104, "h"
   ; 105, "i"
   ; 106, "j"
   ; 107, "k"
   ; 108, "l"
   ; 109, "m"
   ; 110, "n"
   ; 111, "o"
   ; 112, "p"
   ; 113, "q"
   ; 114, "r"
   ; 115, "s"
   ; 116, "t"
   ; 117, "u"
   ; 118, "v"
   ; 119, "w"
   ; 120, "x"
   ; 121, "y"
   ; 122, "z"
   ; 123, "{"
   ; 124, "|"
   ; 125, "}"
   ; 126, "~"
   ; 161, "¡"
   ; 162, "¢"
   ; 163, "£"
   ; 164, "¤"
   ; 165, "¥"
   ; 166, "¦"
   ; 167, "§"
   ; 168, "¨"
   ; 169, "©"
   ; 170, "ª"
   ; 171, "«"
   ; 172, "¬"
   ; 174, "®"
   ; 175, "¯"
   ; 176, "°"
   ; 177, "±"
   ; 178, "²"
   ; 179, "³"
   ; 180, "´"
   ; 181, "µ"
   ; 182, "¶"
   ; 183, "·"
   ; 184, "¸"
   ; 185, "¹"
   ; 186, "º"
   ; 187, "»"
   ; 188, "¼"
   ; 189, "½"
   ; 190, "¾"
   ; 191, "¿"
   ; 192, "À"
   ; 193, "Á"
   ; 194, "Â"
   ; 195, "Ã"
   ; 196, "Ä"
   ; 197, "Å"
   ; 198, "Æ"
   ; 199, "Ç"
   ; 200, "È"
   ; 201, "É"
   ; 202, "Ê"
   ; 203, "Ë"
   ; 204, "Ì"
   ; 205, "Í"
   ; 206, "Î"
   ; 207, "Ï"
   ; 208, "Ð"
   ; 209, "Ñ"
   ; 210, "Ò"
   ; 211, "Ó"
   ; 212, "Ô"
   ; 213, "Õ"
   ; 214, "Ö"
   ; 215, "×"
   ; 216, "Ø"
   ; 217, "Ù"
   ; 218, "Ú"
   ; 219, "Û"
   ; 220, "Ü"
   ; 221, "Ý"
   ; 222, "Þ"
   ; 223, "ß"
   ; 224, "à"
   ; 225, "á"
   ; 226, "â"
   ; 227, "ã"
   ; 228, "ä"
   ; 229, "å"
   ; 230, "æ"
   ; 231, "ç"
   ; 232, "è"
   ; 233, "é"
   ; 234, "ê"
   ; 235, "ë"
   ; 236, "ì"
   ; 237, "í"
   ; 238, "î"
   ; 239, "ï"
   ; 240, "ð"
   ; 241, "ñ"
   ; 242, "ò"
   ; 243, "ó"
   ; 244, "ô"
   ; 245, "õ"
   ; 246, "ö"
   ; 247, "÷"
   ; 248, "ø"
   ; 249, "ù"
   ; 250, "ú"
   ; 251, "û"
   ; 252, "ü"
   ; 253, "ý"
   ; 254, "þ"
   ; 255, "ÿ"
   ; 0, "Ā"
   ; 1, "ā"
   ; 2, "Ă"
   ; 3, "ă"
   ; 4, "Ą"
   ; 5, "ą"
   ; 6, "Ć"
   ; 7, "ć"
   ; 8, "Ĉ"
   ; 9, "ĉ"
   ; 10, "Ċ"
   ; 11, "ċ"
   ; 12, "Č"
   ; 13, "č"
   ; 14, "Ď"
   ; 15, "ď"
   ; 16, "Đ"
   ; 17, "đ"
   ; 18, "Ē"
   ; 19, "ē"
   ; 20, "Ĕ"
   ; 21, "ĕ"
   ; 22, "Ė"
   ; 23, "ė"
   ; 24, "Ę"
   ; 25, "ę"
   ; 26, "Ě"
   ; 27, "ě"
   ; 28, "Ĝ"
   ; 29, "ĝ"
   ; 30, "Ğ"
   ; 31, "ğ"
   ; 32, "Ġ"
   ; 127, "ġ"
   ; 128, "Ģ"
   ; 129, "ģ"
   ; 130, "Ĥ"
   ; 131, "ĥ"
   ; 132, "Ħ"
   ; 133, "ħ"
   ; 134, "Ĩ"
   ; 135, "ĩ"
   ; 136, "Ī"
   ; 137, "ī"
   ; 138, "Ĭ"
   ; 139, "ĭ"
   ; 140, "Į"
   ; 141, "į"
   ; 142, "İ"
   ; 143, "ı"
   ; 144, "Ĳ"
   ; 145, "ĳ"
   ; 146, "Ĵ"
   ; 147, "ĵ"
   ; 148, "Ķ"
   ; 149, "ķ"
   ; 150, "ĸ"
   ; 151, "Ĺ"
   ; 152, "ĺ"
   ; 153, "Ļ"
   ; 154, "ļ"
   ; 155, "Ľ"
   ; 156, "ľ"
   ; 157, "Ŀ"
   ; 158, "ŀ"
   ; 159, "Ł"
   ; 160, "ł"
   ; 173, "Ń"
  |]
;;

exception UnexpectedLine of string

module Tokenizer = struct
  type t =
    { re : Re2.t
    ; encoder : (string, int) Hashtbl.t
    ; decoder : (int, string) Hashtbl.t
    ; bpe_ranks : (string * string, int) Hashtbl.t
    ; start_of_text_token : int
    ; end_of_text_token : int
    }

  let make bpe_path =
    let bpe_lines = Stdio.In_channel.read_lines bpe_path in
    let bpe_lines = Base.List.take bpe_lines (49152 - 256 - 2 + 1) in
    let bpe_lines = List.tl bpe_lines in
    let bpe_lines =
      Base.List.map bpe_lines ~f:(fun line ->
        let vs = Array.of_list @@ Base.String.split ~on:' ' line in
        if Array.length vs != 2
        then
          raise
            (UnexpectedLine
               (Printf.sprintf
                  "UnexpectedLine %d components. %s\n"
                  (Array.length vs)
                  line))
        else vs.(0), vs.(1))
    in
    let vocab1 = Base.Array.map bytes_to_unicode ~f:(fun (_i, elem) -> elem) in
    let vocab2 =
      Base.Array.map bytes_to_unicode ~f:(fun (_i, elem) -> Printf.sprintf "%s</w>" elem)
    in
    let vocab3 = Base.Array.create ~len:(List.length bpe_lines) "" in
    Base.List.iteri bpe_lines ~f:(fun i (l, r) -> vocab3.(i) <- Printf.sprintf "%s%s" l r);
    let vocab = Base.Array.concat [ vocab1; vocab2; vocab3 ] in
    let start_of_text_token = Array.length vocab in
    let vocab = Base.Array.append vocab [| "<|startoftext|>" |] in
    let end_of_text_token = Array.length vocab in
    let vocab = Base.Array.append vocab [| "<|endoftext|>" |] in
    let encoder = Hashtbl.create (Array.length vocab) in
    Base.Array.iteri vocab ~f:(fun i s -> Hashtbl.add encoder s i);
    let decoder = Hashtbl.create (Array.length vocab) in
    Hashtbl.iter (fun k v -> Hashtbl.add decoder v k) encoder;
    let bpe_ranks = Hashtbl.create (List.length bpe_lines) in
    Base.List.iteri bpe_lines ~f:(fun i line -> Hashtbl.add bpe_ranks line i);
    let re = pat in
    { re; encoder; decoder; bpe_ranks; start_of_text_token; end_of_text_token }
  ;;

let encode_pad t s _pad_size_to =
  let s = Base.String.lowercase s in
  (* let bpe_tokens = [t.start_of_text_token] in *)
  let matches = Re2.find_all_exn t.re s in
  Lwt.all @@ Base.List.map matches ~f:(Lwt_log.info)
end

    (* pub fn encode_pad(&self, s: &str, pad_size_to: Option<usize>) -> anyhow::Result<Vec<usize>> { *)

(* let encode t s = *)
(*   () *)
